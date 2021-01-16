import torch 
import torch.nn as nn
import torch.nn.functional as F 

from .layers import WordEmbedding, TanhNgramFeatWithSeq, AddictiveAttention, LastFeat, FM, VariationalDropout
from .layers import UserCoRelLogit, CosineInteraction, WordScore, SingleRelLogit, ZeroAttention
from .utils import attention_weighted_sum


class RelevancMatch(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, kernel_sizes, latent_dim,
                vocab_size, user_size, item_size, rv_len, pretrained_embeddings,
                dropout, pooling_mode, sparse, word_dropout, arch="zero_attention"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_dim = hidden_dim // len(pooling_mode.split("_"))
        self.arch = arch

        self.word_embedding = WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings, 
                                            padding_idx=0)
        self.var_dropout = VariationalDropout(p=word_dropout)
        self.ngram_feat_layer = TanhNgramFeatWithSeq(kernel_sizes, embedding_dim, self.seq_dim, rv_len, pooling_mode)
        
        # relevance coattention 
        self._cosine_interaction = CosineInteraction() 
        self._word_score_layer = WordScore(self.seq_dim)
        self._rel_score_layer = nn.Sequential(nn.Dropout(p=0.1),
                                            nn.Linear(rv_len*2, 16),
                                            nn.Tanh(),
                                            nn.Linear(16, 1))
        self.rel_logit_layer = UserCoRelLogit(self.seq_dim, rv_len, 
                    cosine_interaction=self._cosine_interaction, word_score_layer=self._word_score_layer, 
                    rel_score_layer=self._rel_score_layer)
        self.single_rel_logit_layer = SingleRelLogit(hidden_dim,rv_len, 
                    cosine_interaction=self._cosine_interaction, word_score_layer=self._word_score_layer, 
                    rel_score_layer=self._rel_score_layer)

        # special arch 
        if "zero_attention" in self.arch:
            self.zero_attention = ZeroAttention()
            self.user_zero_embeddings = nn.Embedding(vocab_size, hidden_dim)
            self.item_zero_embeddings = nn.Embedding(vocab_size, hidden_dim)

        # feature 
        self.user_last_feat_layer = LastFeat(user_size, hidden_dim, latent_dim, padding_idx=0)
        self.item_last_feat_layer = LastFeat(item_size, hidden_dim, latent_dim, padding_idx=0)


        self.fm = FM(user_size, item_size, latent_dim, dropout, user_padding_idx=0, item_padding_idx=0)

    def forward(self, u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, u_ids, i_ids, 
                    ui_rev, neg_ui_rev, ui_mask, neg_ui_mask, training):
        bz, rv_num, rv_len = list(u_revs.size())

        # [bz*rv_num, hdim], [bz*rv_num, rv_len, hdim]
        u_revs = self.var_dropout(self.word_embedding(u_revs).view(bz*rv_num, rv_len, self.embedding_dim))
        u_revs, u_rev_seqs = self.ngram_feat_layer(u_revs, u_rev_word_masks.view(bz*rv_num, rv_len))
        i_revs = self.var_dropout(self.word_embedding(i_revs).view(bz*rv_num, rv_len, self.embedding_dim))
        i_revs, i_rev_seqs = self.ngram_feat_layer(i_revs, i_rev_word_masks.view(bz*rv_num, rv_len))

        u_rev_seqs, i_rev_seqs = u_rev_seqs.view(bz, rv_num, rv_len, self.seq_dim), i_rev_seqs.view(bz, rv_num, rv_len, self.seq_dim)
        user_logits  = self.rel_logit_layer(u_rev_seqs, i_rev_seqs, u_rev_word_masks, i_rev_word_masks) #[bz, rv_num]
        item_logits  = self.rel_logit_layer(i_rev_seqs, u_rev_seqs, i_rev_word_masks, u_rev_word_masks) #[bz, rv_num]
        u_revs, i_revs = u_revs.view(bz, rv_num, self.hidden_dim), i_revs.view(bz, rv_num, self.hidden_dim)
        if "zero_attention" in self.arch:
            u_feat, user_logits = self.zero_attention(u_revs, u_rev_masks, user_logits, self.user_zero_embeddings(u_ids))
            i_feat, item_logits = self.zero_attention(i_revs, i_rev_masks, item_logits, self.item_zero_embeddings(i_ids))
        else:
            user_scores = F.softmax(torch.masked_fill(user_logits, ~u_rev_masks, -1e8), dim=1)
            item_scores = F.softmax(torch.masked_fill(item_logits, ~i_rev_masks, -1e8), dim=1)

            u_feat = attention_weighted_sum(user_scores, u_revs)
            i_feat = attention_weighted_sum(item_scores, i_revs)
        
        if training:
            ui_rev = self.ngram_feat_layer(self.var_dropout(self.word_embedding(ui_rev)), ui_mask)[1]
            neg_ui_rev = self.ngram_feat_layer(self.var_dropout(self.word_embedding(neg_ui_rev)), neg_ui_mask)[1]
            ui_logits = self.single_rel_logit_layer(ui_rev, i_rev_seqs, ui_mask, i_rev_word_masks)
            neg_ui_logits = self.single_rel_logit_layer(neg_ui_rev, i_rev_seqs, neg_ui_mask, i_rev_word_masks)

        u_feat = self.user_last_feat_layer(u_feat, u_ids)
        i_feat = self.item_last_feat_layer(i_feat, i_ids) #[bz, hdim]

        preds = self.fm(u_feat, i_feat, u_ids, i_ids)

        if training:
            return preds.view(bz), (ui_logits.view(bz), neg_ui_logits.view(bz)), (user_logits, item_logits)
        else:
            return preds.view(bz), (user_logits, item_logits)

class SiameseCNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, kernel_sizes, latent_dim,
                vocab_size, user_size, item_size, rv_len,
                pretrained_embeddings, dropout, pooling_mode, sparse, word_dropout):
        super().__init__() 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        out_feature_per_kernel = int(hidden_dim / (len(pooling_mode.split("_")) * len(kernel_sizes)))
        print("out feature per kernel: ", out_feature_per_kernel)

        self.word_embedding = WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings,
                                            padding_idx=0, sparse=sparse)
        self.var_dropout = VariationalDropout(p=word_dropout)
        
        self.ngram_feat_layer =  TanhNgramFeatWithSeq(kernel_sizes, embedding_dim, out_feature_per_kernel, rv_len, pooling_mode)

        self.review_att_layer = AddictiveAttention(hidden_dim, latent_dim)

        self.user_last_feat_layer = LastFeat(user_size, hidden_dim, latent_dim, padding_idx=0)
        self.item_last_feat_layer = LastFeat(item_size, hidden_dim, latent_dim, padding_idx=0)

        self.fm = FM(user_size, item_size, latent_dim, dropout, user_padding_idx=0, item_padding_idx=0)

    def forward(self, u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, u_ids, i_ids):
        """
        Args:
            u_revs: [bz, rv_num, rv_len]
            i_revs:
            u_rev_word_masks
            i_rev_word_masks
            u_rev_masks: [bz, rv_num]
            i_rev_masks: [bz, rv_num]
            u_ids: [bz]
            i_ids: [bz]
        
        Returns:
            out_logits: [bz]
            u_rev_scores: [bz, rv_num]
            i_rev_scores: [bz, ]
        """
        bz, rv_num, rv_len = list(u_revs.size())

        # each review representation
        u_revs = self.var_dropout(self.word_embedding(u_revs).view(bz*rv_num, rv_len, self.embedding_dim))
        u_revs = self.ngram_feat_layer(u_revs, u_rev_word_masks.view(bz*rv_num, rv_len))[0].view(bz, rv_num, self.hidden_dim)
        i_revs = self.var_dropout(self.word_embedding(i_revs).view(bz*rv_num, rv_len, self.embedding_dim))
        i_revs = self.ngram_feat_layer(i_revs, i_rev_word_masks.view(bz*rv_num, rv_len) )[0].view(bz, rv_num, self.hidden_dim)
        
        # user/item representation 
        u_rev_feat, u_rev_scores = self.review_att_layer(u_revs, u_rev_masks)
        i_rev_feat, i_rev_scores = self.review_att_layer(i_revs, i_rev_masks)

        # user/item combine representation 
        u_feat = self.user_last_feat_layer(u_rev_feat, u_ids)
        i_feat = self.item_last_feat_layer(i_rev_feat, i_ids) #[bz, hdim]

        # fm 
        out_logits = self.fm(u_feat, i_feat, u_ids, i_ids)

        return out_logits.view(bz), u_rev_scores, i_rev_scores
