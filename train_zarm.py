import pickle 
import argparse
import json
import os
import time
import re
from collections import defaultdict
import csv
import gzip
import math
import random

import torch 
import torch.nn as nn
from torch import LongTensor, FloatTensor
import numpy as np
from gensim.models import KeyedVectors

from experiment import Experiment
from utils import get_mask
from data.divide_and_create_examples import clean_str

np.set_printoptions(precision=3)

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op 
        self.param_groups = self.optimizers[-1].param_groups
    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()
    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        list_of_state_dict = []
        for op in self.optimizers:
            list_of_state_dict.append(op.state_dict())
        return list_of_state_dict

class MultipleScheduler(object):
    def __init__(self, Scheduler, *ops, **kwargs):
        self._optimizers = ops
        self._schedulers =  [Scheduler(optim, **kwargs) for optim in self._optimizers]
    
    def step(self, val):
        for sl in self._schedulers:
            sl.step(val)

class Args(object):
    pass

def parse_args(config):
    args = Args()
    with open(config, 'r') as f:
        config = json.load(f)
    for name, val in config.items():
        setattr(args, name, val)

    return args

def load_pretrained_embeddings(vocab, word2vec, emb_size):
    """
    NOTE:
        tensorflow version.
    Args:
        vocab: a Vocab object
        word2vec: dictionry, (str, np.ndarry with type of np.float32)

    Return:
        pre_embeddings: torch.FloatTensor
    """
    pre_embeddings = np.random.uniform(-1.0, 1.0, size=[len(vocab), emb_size]).astype(np.float32)
    for word in vocab._token2id:
        if word in word2vec:
            pre_embeddings[vocab._token2id[word]] = word2vec[word]
    return torch.FloatTensor(pre_embeddings)

class AvgMeters(object):
    def __init__(self):
        self.count = 0
        self.total = 0. 
        self._val = 0.
    
    def update(self, val, count=1):
        self.total += val
        self.count += count

    def reset(self):
        self.count = 0
        self.total = 0. 
        self._val = 0.

    @property
    def val(self):
        return self.total / self.count

class EarlyStop(Exception):
    pass

class NarreExperiment(Experiment):
    def __init__(self, args, dataloaders):
        super(NarreExperiment, self).__init__(args, dataloaders)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dataloader
        self.train_dataloader = dataloaders["train"]
        self.valid_dataloader = dataloaders["valid"] if dataloaders["valid"] is not None else None

        # stats
        self.train_stats = defaultdict(list)
        self.valid_stats = defaultdict(list)
        self._best_rmse = 1e3
        self.patience = 0

        # create output path
        self.setup()
        self.build_model() # self.model
        self.build_optimizer() #self.optimizer
        self.build_scheduler() #self.scheduler
        self.build_loss_func() #self.loss_func

        # print
        self.print_args()
        self.print_model_stats()

    def build_scheduler(self):
        if self.args.sparse:
            self.scheduler = MultipleScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau, self.sparse_optim, self.dense_optim,
                                            mode="min", factor=0.5, patience=0)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=0)

    def parse_kernel_sizes(self, str_kernel_sizes):
        kernel_sizes = [int(x) for x in str_kernel_sizes.split(",")]
        print("kernel sizes are: ", kernel_sizes)
        return kernel_sizes
        
    def build_model(self):
        # import different model 
        if self.args.model == "relevance_highway":
            from models.relevance_cnn_highway import RelevanceCNNHighway as RelevanceCNN
            self.print_write_to_log("Use RelevanceCNNHighway ...")
        elif self.args.model == "relevance_highway_coattention":
            from models.relevance_cnn_highway_coattention import RelevanceCNNHighwayCoAttention as RelevanceCNN
            self.print_write_to_log("Use RelevanceCNNHighwayCoAttention...")
        elif self.args.model == "relevance_match":
            from models.relevance_cnn import RelevancMatch as RelevanceCNN
            self.print_write_to_log("Use RelevanceMatch ...")
        elif self.args.model == "siamese":
            from models.relevance_cnn import SiameseCNN as SiameseCNN
        else:
            raise ValueError(f"model: {self.args.model} not found")

        # word pretrain
        if self.args.use_pretrain:
            data_prefix = "/raid/hanszeng/Recommender/NARRE/data/"
            pretrain_path = "GoogleNews-vectors-negative300.bin"
            pretrain_path = data_prefix + pretrain_path

            
            wv_from_bin = KeyedVectors.load_word2vec_format(pretrain_path, binary=True)

            word2vec = {}
            for word, vec in zip(wv_from_bin.vocab, wv_from_bin.vectors):
                word2vec[word] = vec
            
            
            _dataset = self.train_dataloader.dataset
            word_pretrained = load_pretrained_embeddings(_dataset.word_vocab, word2vec, self.args.embedding_dim)
        else:
            _dataset  = self.train_dataloader.dataset
            word_pretrained=None
        kernel_sizes = self.parse_kernel_sizes(self.args.kernel_sizes)

        # model initialization
        if "relevance" in self.args.model:
            """
            self.model = RelevanceCNN(embedding_dim=self.args.embedding_dim, hidden_dim=self.args.hidden_dim, kernel_sizes=kernel_sizes,
                                latent_dim=self.args.latent_dim, vocab_size=len(_dataset.word_vocab), 
                                user_size=_dataset.user_num, item_size=_dataset.item_num,  
                                rv_len=_dataset.rv_len, pretrained_embeddings=word_pretrained, 
                                dropout=self.args.dropout, pooling_mode=self.args.pooling, sparse=self.args.sparse, 
                                word_dropout=self.args.word_dropout, mode=self.args.mode, temperature=self.args.temperature,
                                arch=self.args.arch, use_ui_bias=self.args.use_ui_bias)
            """
            self.model = RelevanceCNN(embedding_dim=self.args.embedding_dim, hidden_dim=self.args.hidden_dim, kernel_sizes=kernel_sizes,
                                latent_dim=self.args.latent_dim, vocab_size=len(_dataset.word_vocab), 
                                user_size=_dataset.user_num, item_size=_dataset.item_num,  
                                rv_len=_dataset.rv_len, pretrained_embeddings=word_pretrained, 
                                dropout=self.args.dropout, pooling_mode=self.args.pooling, sparse=self.args.sparse, 
                                word_dropout=self.args.word_dropout, mode=self.args.mode,
                                arch=self.args.arch)
        elif "siamese" in self.args.model:
            self.model = SiameseCNN(embedding_dim=self.args.embedding_dim, hidden_dim=self.args.hidden_dim, kernel_sizes=kernel_sizes,
                                latent_dim=self.args.latent_dim, vocab_size=len(_dataset.word_vocab), 
                                user_size=_dataset.user_num, item_size=_dataset.item_num,  
                                rv_len=_dataset.rv_len, pretrained_embeddings=word_pretrained, 
                                dropout=self.args.dropout, pooling_mode=self.args.pooling, sparse=self.args.sparse, 
                                word_dropout=self.args.word_dropout)
        else:
            raise ValueError("model not found.")
        if self.args.parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.print_write_to_log("the model is parallel training.")
        self.model.to(self.device)

    def build_optimizer(self):
        def get_sparse_and_dense_parameters(model):
            sparse_params = []
            dense_params = []
            for name, params in model.named_parameters():
                if name == "word_embedding.embedding.weight":
                    sparse_params.append(params)
                else:
                    dense_params.append(params)
            print(f"len of params, sparse params, dense params: {len(model.state_dict())}, {len(sparse_params)}, {len(dense_params)}")
            return sparse_params, dense_params

        if self.args.sparse:
            sparse_params, dense_params = get_sparse_and_dense_parameters(self.model)

            self.sparse_optim = torch.optim.SparseAdam(sparse_params, lr=self.args.lr)
            self.dense_optim = torch.optim.Adam(dense_params, lr=self.args.lr)
        
            self.optimizer = MultipleOptimizer(self.sparse_optim, self.dense_optim)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.verbose:
            self.print_write_to_log(re.sub(r"\n", "", self.optimizer.__repr__()))
        
    def build_loss_func(self):
        self.loss_func = nn.MSELoss()
        self.bce_loss_func = nn.BCEWithLogitsLoss()



    def train_one_epoch(self, current_epoch):
        avg_cls_loss = AvgMeters()
        avg_mse_loss = AvgMeters()
        square_error = 0.
        accum_count = 0
        start_time = time.time()

        self.model.train()
        for i,  ((u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), (u_rids, i_rids), \
                (ui_revs, neg_ui_revs, ui_word_masks, neg_ui_word_masks, ui_labels, neg_ui_labels)) in enumerate(self.train_dataloader):
            if i == 0 and current_epoch == 0:
                print("u_text", u_revs.shape, "i_text", i_revs.shape, "reuid", u_rids.shape, "reiid", i_rids.shape)
            u_ids = u_ids.to(self.device)
            i_ids = i_ids.to(self.device)
            ratings = ratings.to(self.device)
            u_revs = u_revs.to(self.device)
            i_revs = i_revs.to(self.device)
            u_rev_word_masks = u_rev_word_masks.to(self.device)
            i_rev_word_masks = i_rev_word_masks.to(self.device)
            u_rev_masks = u_rev_masks.to(self.device)
            i_rev_masks = i_rev_masks.to(self.device)
            
            ui_revs = ui_revs.to(self.device)
            neg_ui_revs = neg_ui_revs.to(self.device)
            ui_word_masks = ui_word_masks.to(self.device)
            neg_ui_word_masks = neg_ui_word_masks.to(self.device)
            ui_labels = ui_labels.to(self.device)
            neg_ui_labels = neg_ui_labels.to(self.device)

            self.optimizer.zero_grad()
            if "relevance" in self.args.model:
                y_pred, (gt_logits, ng_logits), (rel_feat_norm, zero_perc) = \
                                    self.model(u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, 
                                                u_ids, i_ids, ui_revs, neg_ui_revs, ui_word_masks, neg_ui_word_masks, training=True)
                
                loss_1 = self.loss_func(y_pred, ratings)
                loss_2 = self.bce_loss_func(gt_logits, ui_labels) + self.bce_loss_func(ng_logits, neg_ui_labels)
                loss = loss_1 + self.args.reg_weight * loss_2 * 0.1
                loss.backward()
            elif "siamese" in self.args.model:
                y_pred, _, _ = self.model(u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, 
                                                u_ids, i_ids)
                loss = self.loss_func(y_pred, ratings)
                loss.backward()
            else:
                raise ValueError("model not found")

            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # val 
            if "relevance" in self.args.model:
                avg_mse_loss.update(loss_1.mean().item())
                square_error += loss_1.mean().item() * ratings.size(0)
                accum_count += ratings.size(0)
                avg_cls_loss.update(loss_2.mean().item())
            elif "siamese" in self.args.model:
                avg_mse_loss.update(loss.mean().item())
                square_error += loss.mean().item() * ratings.size(0)
                accum_count += ratings.size(0)
            else:
                raise ValueError("model not found")

            # log
            if (i+1) % self.args.log_idx == 0 and self.args.log:
                elpased_time = (time.time() - start_time) / self.args.log_idx
                rmse = math.sqrt(square_error / accum_count)

                if "relevance" in self.args.model:
                    
                    log_text = "epoch: {}/{}, step: {}/{}, loss mse: {:.3f}, loss cls: {:.3f}, rmse: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
                        current_epoch, self.args.epochs,  (i+1), len(self.train_dataloader), avg_mse_loss.val, avg_cls_loss.val, rmse, 
                        self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
                    )
                    """
                    log_text = "epoch: {}/{}, step: {}/{}, loss mse: {:.3f}, rmse: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
                        current_epoch, self.args.epochs,  (i+1), len(self.train_dataloader), avg_mse_loss.val, rmse, 
                        self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
                    )
                    """
                    print("rel norm", np.quantile(rel_feat_norm.cpu().data.numpy(), [0.1, 0.5, 0.9]))
                    print("zero perc, ", np.quantile(zero_perc.cpu().data.numpy(), [0.1, 0.5, 0.9]))
                    
                #print("user logits: ", user_logits.cpu().data[:10, :])
                #print("item logits: ", item_logits.cpu().data[:10, :])
                elif "siamese" in self.args.model:
                    log_text = "epoch: {}/{}, step: {}/{}, loss mse: {:.3f}, rmse: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
                        current_epoch, self.args.epochs,  (i+1), len(self.train_dataloader), avg_mse_loss.val, rmse, 
                        self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
                    )
                else:
                    raise ValueError("model not found")
                self.print_write_to_log(log_text)

                avg_mse_loss.reset()
                avg_cls_loss.reset()
                square_error = 0. 
                accum_count = 0
                start_time = time.time()

    def valid_one_epoch(self):
        square_error = 0.
        accum_count = 0
        avg_loss = AvgMeters()

        self.model.eval()
        for i,  ((u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), \
                 (u_rids, i_rids)) in enumerate(self.valid_dataloader):
            u_ids = u_ids.to(self.device)
            i_ids = i_ids.to(self.device)
            ratings = ratings.to(self.device)
            u_revs = u_revs.to(self.device)
            i_revs = i_revs.to(self.device)
            u_rev_word_masks = u_rev_word_masks.to(self.device)
            i_rev_word_masks = i_rev_word_masks.to(self.device)
            u_rev_masks = u_rev_masks.to(self.device)
            i_rev_masks = i_rev_masks.to(self.device)

            with torch.no_grad():
                if "relevance" in self.args.model:
                    y_pred, _= self.model(u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, 
                                    u_ids, i_ids, ui_rev=None, neg_ui_rev=None, ui_mask=None, neg_ui_mask=None, training=False)
                elif "siamese" in self.args.model:
                    y_pred, _, _= self.model(u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, 
                                    u_ids, i_ids)
                else:
                    raise ValueError("model not found")
                loss = self.loss_func(y_pred, ratings)

            square_error += loss.mean().item() * ratings.size(0)
            accum_count += ratings.size(0)
            avg_loss.update(loss.mean().item())

        rmse = math.sqrt(square_error / accum_count)
        if rmse < self.best_rmse:
            self.best_rmse =  rmse 
            self.save("best_model.pt")
            self.patience = 0
        else:
            self.patience += 1

        log_text =  "valid loss: {:.3f}, valid rmse: {:.3f}, best rmse: {:.3f}".format(avg_loss.val, rmse, self.best_rmse)
        self.print_write_to_log(log_text)

        # ealry stop
        if self.patience >= self.args.patience:
            # write stats 
            if self.args.stats:
                self.write_stats("train")
                self.write_stats("valid")

            raise EarlyStop("early stop")
        
        if self.args.use_scheduler:
            self.scheduler.step(rmse)

    @property
    def best_rmse(self):
        return self._best_rmse
    
    @best_rmse.setter
    def best_rmse(self, val):
        self._best_rmse = val

    def train(self):
        print("start training ...")
        for epoch in range(self.args.epochs):
            self.valid_one_epoch()
            self.train_one_epoch(epoch)

class NarreDatasetSameUIReviewNum(torch.utils.data.Dataset):
    def __init__(self, args, set_name):
        super(NarreDatasetSameUIReviewNum, self).__init__()

        self.args = args
        self.set_name = set_name
        param_path = os.path.join(self.args.data_dir, "meta.pkl")
        with open(param_path, "rb") as f:
            para = pickle.load(f)

        self.user_num = para['user_num']
        self.item_num = para['item_num']
        self.indexlizer = para['indexlizer']
        self.rv_num = para["rv_num"]
        self.rv_len = para["rv_len"]
        self.u_text = para['user_reviews']
        self.i_text = para['item_reviews']
        self.u_rids = para["user_rids"]
        self.i_rids = para["item_rids"]
        self.word_vocab = self.indexlizer._vocab

        example_path = os.path.join(self.args.data_dir, f"{set_name}_exmaples.pkl")
        with open(example_path, "rb") as f:
            self.examples = pickle.load(f)

    def __getitem__(self, i):
        # for each review(u_text or i_text) [...] 
        # NOTE: not padding 
        if self.set_name == "train":
            u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, _, _, ui_rev = self.examples[i]
            
            neg_idx = random.randint(0, len(self.examples)-1) 
            while self.examples[neg_idx][1] == i_id:
                neg_idx = random.randint(0, len(self.examples)-1)
            neg_ui_rev = self.examples[neg_idx][-1]

            p = random.random()
            if p <= self.args.random_zero_example_rate:
                neg_ui_rev = self.rv_len * [0]
            if p <= self.args.random_zero_example_rate / 2.:
                ui_rev = self.rv_len * [0]

            ui_label = 1. 
            neg_ui_label = 0.

            return u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, ui_rev, neg_ui_rev, ui_label, neg_ui_label      

        else:
            u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, _, _ = self.examples[i]
            return u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids
        
    def __len__(self):
        return len(self.examples)

    @staticmethod
    def truncate_tokens(tokens, max_seq_len):
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        return tokens

    @staticmethod
    def get_rev_mask(inputs):
        """
        If rv_len are all 0, then corresponding position in rv_num should be 0
        Args:
            inputs: [bz, rv_num, rv_len]
        """
        bz, rv_num, _ = list(inputs.size())

        masks = torch.ones(size=(bz, rv_num)).int()
        inputs = inputs.sum(dim=-1) #[bz, rv_num]
        masks[inputs==0] = 0 

        return masks.bool()

    def train_collate_fn(self, batch):
        u_ids, i_ids, ratings, u_revs, i_revs, u_rids, i_rids, ui_revs, neg_ui_revs, ui_labels, neg_ui_labels = zip(*batch)
        u_ids = LongTensor(u_ids)
        i_ids = LongTensor(i_ids)
        ratings = FloatTensor(ratings)
        u_revs = LongTensor(u_revs)
        i_revs = LongTensor(i_revs)
        u_rids = LongTensor(u_rids)
        i_rids = LongTensor(i_rids)
        ui_revs = LongTensor(ui_revs) 
        neg_ui_revs = LongTensor(neg_ui_revs)
        ui_labels = FloatTensor(ui_labels)
        neg_ui_labels = FloatTensor(neg_ui_labels)

        u_rev_word_masks = get_mask(u_revs)
        i_rev_word_masks = get_mask(i_revs)
        ui_word_masks = get_mask(ui_revs)
        neg_ui_word_masks = get_mask(neg_ui_revs)
        u_rev_masks = self.get_rev_mask(u_revs)
        i_rev_masks = self.get_rev_mask(i_revs)

        return (u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), (u_rids, i_rids), \
                (ui_revs, neg_ui_revs, ui_word_masks, neg_ui_word_masks,  ui_labels, neg_ui_labels)

    def test_collate_fn(self, batch):
        u_ids, i_ids, ratings, u_revs, i_revs, u_rids, i_rids = zip(*batch)
        
        u_ids = LongTensor(u_ids)
        i_ids = LongTensor(i_ids)
        ratings = FloatTensor(ratings)
        u_revs = LongTensor(u_revs)
        i_revs = LongTensor(i_revs)
        u_rids = LongTensor(u_rids)
        i_rids = LongTensor(i_rids)

        u_rev_word_masks = get_mask(u_revs)
        i_rev_word_masks = get_mask(i_revs)
        u_rev_masks = self.get_rev_mask(u_revs)
        i_rev_masks = self.get_rev_mask(i_revs)

        return (u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), (u_rids, i_rids)
     

if __name__ == "__main__":
    config_file = "./models/default_train.json"
    args = parse_args(config_file)
    train_dataset = NarreDatasetSameUIReviewNum(args, "train")
    valid_dataset = NarreDatasetSameUIReviewNum(args, "valid")

    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.train_collate_fn, num_workers=2)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.test_collate_fn, num_workers=2)

    dataloaders = {"train": train_dataloder, "valid": valid_dataloader, "test": None}
    experiment = NarreExperiment(args, dataloaders)
    experiment.train()