import torch.nn as nn 
import torch 
import torch.nn.functional as  F

from .utils import masked_tensor, masked_colwise_mean, attention_weighted_sum

# ==== Intercation ====
class ZeroAttention(nn.Module):
    def __init__(self):
        # NOTE: test !!!
        super().__init__()
    
    def forward(self, rev_feats, rev_masks, rev_logits, global_feat=None):
        """
        Args:
            rev_feats: [bz, rv_num, hdim]
            rev_masks: [bz, rv_num]
            global_feat: [bz, hdim]
            rev_logits: [bz, rnum]

        Return:
            combine_feat: [bz, hdim]
        """
        bz, rv_num, hdim = rev_feats.size(0), rev_feats.size(1), rev_feats.size(2)
        _device = rev_feats.device
        if global_feat is None:
            global_feat = torch.zeros(bz, hdim).to(_device)

        global_masks = torch.ones(size=(bz, 1)).bool().to(_device)
        global_logits = 0.5*torch.ones(size=(bz, 1)).float().to(_device)

        rev_masks = torch.cat([rev_masks, global_masks], dim=-1) #[bz, rv_num+1]
        rev_logits = torch.cat([rev_logits, global_logits], dim=-1) #[bz, rv_num+1]
        rev_scores = F.softmax(torch.masked_fill(rev_logits, ~rev_masks, -1e8), dim=-1).view(bz, rv_num+1, 1) 

        global_feat = global_feat.view(bz, 1, hdim)
        combine_feat = torch.cat([rev_feats, global_feat], dim=1) #[bz, rv_num+1, hdim]

        combine_feat = torch.sum(combine_feat * rev_scores, dim=1) #[bz, hdim]

        return combine_feat, rev_scores.view(bz, rv_num+1)
        




class CosineInteraction(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, input1, input2):
        """
        input1: [*, seq_len_a, dim]
        input2: [*, seq_len_b, dim]
        """
        device = input1.device

        input1_norm = torch.norm(input1, p=2, dim=-1, keepdim=True)
        input2_norm = torch.norm(input2, p=2, dim=-1, keepdim=True) #[*, seq_len_b, 1]

        _y = torch.bmm(input1, input2.transpose(1,2)) # [*, seq_len_a, seq_len_b]
        _y_norm = torch.bmm(input1_norm, input2_norm.transpose(1,2)) #[*, seq_len_a, seq_len_b]
        epilson = torch.tensor(1e-6).to(device)

        return _y / torch.max(_y_norm, epilson)

class WordScore(nn.Module):
    def __init__(self, feature):
        super().__init__()

        self.inner_product_layer = nn.Linear(feature, 1, bias=False)
        #self.inner_product_layer = nn.Sequential(nn.Linear(feature, 1))
    def forward(self, inputs, masks):
        """
        Args:
            inputs: [bz, seq_len, dim]
            masks: [bz, seq_len]

        Returns:
            out_scores: [bz, seq_len]
        """
        bz, seq_len, dim = list(inputs.size())
        inputs = self.inner_product_layer(inputs).view(bz, seq_len) #[bz, seq_len]
        assert inputs.dim() == masks.dim()

        masked_inputs = torch.masked_fill(inputs, ~masks, -1e8)
        out_scores = F.softmax(masked_inputs, dim=-1)

        return out_scores

class VariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):

        """
        Apply dropout to input tensor.
        # Parameters
        input_tensor : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`
        # Returns
        output : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor
            
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, padding_idx=0, freeze_embeddings=False, sparse=False):
        super(WordEmbedding, self).__init__()

        self.freeze_embeddings = freeze_embeddings

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, sparse=sparse)
        self.embedding.weight.requires_grad = not self.freeze_embeddings
        if pretrained_embeddings is not None:
            self.embedding.load_state_dict({"weight": torch.tensor(pretrained_embeddings)})
        else:
            print("[Warning] not use pretrained embeddings ...")

    def forward(self, inputs):
        out = self.embedding(inputs)     
        return out

class MaskedAvgPooling1d(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, hdim, seq_len]
            input_masks: [bz, seq_len]

        Returns: 
            outputs: [bz, hdim, 1]
        """
        assert input_masks.dim() == 2
        float_masks = input_masks.unsqueeze(1).float() #[bz, 1, seq_len]
        input_lengths = float_masks.sum(dim=2, keepdim=True) + 1e-8  #[bz, 1, 1]
        #print("inputs: ", inputs)
        #print("input_lengths: ", input_lengths)
        sum_inputs = (inputs * float_masks).sum(dim=2, keepdim=True) #[bz, hdim, 1]
        #print(sum_inputs)
        return sum_inputs / input_lengths

class TanhNgramFeatWithSeq(nn.Module):
    def __init__(self, kernel_sizes, in_feature, out_feature_per_kernel, seq_len, mode="MAX_AVG"):
        super().__init__()
        assert all([ type(x)==int for x in kernel_sizes])
        self.out_feature_per_kernel = out_feature_per_kernel
        self.seq_len = seq_len
        self.mode = mode 

        self.ngrams = nn.ModuleList([nn.Sequential(nn.Conv1d(in_feature, out_feature_per_kernel, kz), nn.Tanh()) 
                        for kz in kernel_sizes])
        self.kernel_sizes = kernel_sizes
        
        if "AVG" in self.mode:
            self.masked_avgpool_1d = MaskedAvgPooling1d()
            print("use avg pooling")
        if "MAX" in self.mode:
            print('use maxpooling')
    
    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, in_feat]
            input_masks: [bz, seq_len]

        Returns:
            out_ngrams: [bz, out_feat]
            ngram_seqs: [bz, seq_len, out_feat//2]
        """
        bz, seq_len, _ = list(inputs.size())
        inputs = masked_tensor(inputs, input_masks)
        inputs = inputs.transpose(1,2)
        list_of_ngram = [ng_layer(inputs) for ng_layer in self.ngrams] # list of [bz, out_feat, var_seq_lens]
        #print("list of ngram: ", list_of_ngram)
        out_ngrams = []

        if "MAX" in self.mode:
            #print("use maxpooling")
            list_max_ngrams = [F.max_pool1d(ngram, self.seq_len-kz+1) for ngram, kz in zip(list_of_ngram, self.kernel_sizes)] # list of [bz, out_feat, 1]
            out_ngrams += list_max_ngrams
        if "AVG" in self.mode:
            #print("use avgpooling")
            list_avg_masks = [input_masks[:, :self.seq_len-kz+1]  for kz in self.kernel_sizes]
            list_avg_ngrams = [self.masked_avgpool_1d(ngram, mask) for ngram, mask in  zip(list_of_ngram, list_avg_masks)] # list of [bz, out_feat, 1]
            out_ngrams += list_avg_ngrams

        out_ngrams = torch.cat(out_ngrams, dim=1).view(bz, -1)
        assert len(list_of_ngram) == 1
        ngram_seqs = list_of_ngram[0].transpose(1,2).contiguous()

        return out_ngrams, ngram_seqs


class TanhNgramFeatWithSeqOnlyOneKernel(nn.Module):
    def __init__(self, kernel_size, in_feature, out_feature, seq_len, mode="MAX"):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.seq_len = seq_len
        self.mode = mode 

        self.ngrams = nn.Sequential(nn.Conv1d(in_feature, out_feature, kernel_size), 
                                    nn.Tanh())
        if "AVG" == self.mode:
            self.masked_avgpool_1d = MaskedAvgPooling1d()
            print("use avg pooling")
        elif "MAX" in self.mode:
            print('use maxpooling')
        else:
            raise ValueError(f"pooling technique {mode} is not reasonable.")
    
    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, in_feat]
            input_masks: [bz, seq_len]

        Returns:
            ngram_aggre: [bz, out_feature]
            ngrams: [bz, seq_len, out_feature]
        """
        bz, _, _ = list(inputs.size())
        inputs = masked_tensor(inputs, input_masks)
        inputs = inputs.transpose(1,2)
        
        ngrams = self.ngrams(inputs)

        if self.mode == "MAX":
            #print("use maxpooling")
            ngram_aggre = F.max_pool1d(ngrams, self.seq_len-self.kernel_size+1)
        if "AVG" in self.mode:
            #print("use avgpooling")
            ngram_aggre = self.masked_avgpool_1d(ngrams, input_masks)

        return ngram_aggre, ngrams
class AddictiveAttention(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()

        self.proj_layer = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                        nn.Tanh())
        self.inner_product = nn.Linear(latent_dim, 1, bias=False)
    
    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, hdim]
            input_masks: [bz, seq_len]
        
        Returns:
            Outputs: [bz, hdim]
        """
        bz, seq_len, hdim = list(inputs.size())

        att_logtis = self.inner_product(self.proj_layer(inputs)) #[bz, seq_len, 1]
        input_masks = input_masks.unsqueeze(2) #[bz, seq_len, 1]
        att_scores = F.softmax(torch.masked_fill(att_logtis, ~input_masks, -1e8), dim=1) #[bz, seq_len, 1]

        outptus = torch.sum(att_scores * inputs, dim=1)

        return outptus, att_scores

class LastFeat(nn.Module):
    def __init__(self, vocab_size, feat_size, latent_dim, padding_idx):
        super(LastFeat, self).__init__()

        self.W = nn.Parameter(torch.Tensor(feat_size, latent_dim))
        self.b = nn.Parameter(torch.Tensor(latent_dim))

        self.ebd = nn.Embedding(vocab_size, latent_dim, padding_idx=padding_idx)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.1
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.constant_(self.b, 0.)
        nn.init.uniform_(self.ebd.weight, -bound, bound)


    def forward(self, text_feat, my_id):
        """
        Args:
            text_feat: [bz, feat_size]
            my_id: [bz]
        """

        out_feat = text_feat @ self.W + self.b + self.ebd(my_id)

        return out_feat

class FMWithoutUIBias(nn.Module):
    def __init__(self, user_size, item_size, latent_dim, dropout, user_padding_idx, item_padding_idx):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.h = nn.Parameter(torch.Tensor(latent_dim, 1))
        self.g_bias = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.1
        nn.init.uniform_(self.h, -bound, bound)
        nn.init.constant_(self.g_bias, 4.0)


    def forward(self, u_feat, i_feat):
        """
        Args:
            u_feat: [bz, latent_dim]
            i_feat: ...
            u_id: [bz]
            i_id: [bz]

        Returns:
            pred: [bz]
        """
        fm = torch.mul(u_feat, i_feat)
        fm = F.relu(fm)
        fm = self.dropout(fm) #[bz, latent_dim]

        pred = fm @ self.h + self.g_bias

        return pred

class FM(nn.Module):
    def __init__(self, user_size, item_size, latent_dim, dropout, user_padding_idx, item_padding_idx):
        super(FM, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.h = nn.Parameter(torch.Tensor(latent_dim, 1))

        self.user_bias = nn.Embedding(user_size, 1, padding_idx=user_padding_idx)
        self.item_bias = nn.Embedding(item_size, 1, padding_idx=item_padding_idx)
        self.g_bias = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.1
        nn.init.uniform_(self.h, -bound, bound)
        nn.init.uniform_(self.user_bias.weight, -bound, bound)
        nn.init.uniform_(self.item_bias.weight, -bound, bound)
        nn.init.constant_(self.g_bias, 4.0)


    def forward(self, u_feat, i_feat, u_id, i_id):
        """
        Args:
            u_feat: [bz, latent_dim]
            i_feat: ...
            u_id: [bz]
            i_id: [bz]

        Returns:
            pred: [bz]
        """
        fm = torch.mul(u_feat, i_feat)
        fm = F.relu(fm)
        fm = self.dropout(fm) #[bz, latent_dim]

        u_bias = self.user_bias(u_id)
        i_bias = self.item_bias(i_id)

        pred = fm @ self.h + u_bias + i_bias + self.g_bias

        return pred

class UserCoRelLogit(nn.Module):
    def __init__(self, hidden_dim, rv_len, cosine_interaction, word_score_layer, rel_score_layer):
        super().__init__() 

        self.hidden_dim = hidden_dim

        self.cosine_interaction = cosine_interaction
        self.word_score_layer = word_score_layer
        self.rel_score_layer = rel_score_layer

    def forward(self, seq_a, seq_b, mask_a, mask_b):
        """
        Args:
            seq_a: [bz, rv_num, rv_length, dim]
            seq_b: [bz, rv_num, rv_length, dim]
            mask_a: [bz, rv_num, rv_length]
            mask_b: [bz, rv_num, rv_length]

        Returns:
           ab_logits: [bz, rv_num]
        """
        bz, rv_num, rv_len, hdim = list(seq_a.size())
        assert hdim == self.hidden_dim

        expand_seq_b = seq_b.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        expand_mask_b = mask_b.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, 1, rv_num*rv_len)
        seq_a = seq_a.view(bz*rv_num, rv_len, hdim)
        mask_a = mask_a.view(bz*rv_num, rv_len)
        
        atob_cos_affinity = self.cosine_interaction(seq_a, expand_seq_b) #[*, rv_len, rv_num*rv_len]
        ab_mean_feat = masked_colwise_mean(atob_cos_affinity, expand_mask_b) #[*, rv_len, 1]
        ab_max_feat, _ = atob_cos_affinity.max(dim=-1, keepdim=True) #[*, rv_len, 1]
        ab_feat = torch.cat([ab_mean_feat, ab_max_feat], dim=-1).view(bz*rv_num, rv_len, 2)
        ab_word_score = self.word_score_layer(seq_a, mask_a).view(bz*rv_num, rv_len, 1)
        ab_feat = (ab_feat * ab_word_score).view(bz*rv_num, rv_len*2)
        ab_logits = self.rel_score_layer(ab_feat).view(bz, rv_num) # [bz, rv_num]
        
        return ab_logits

class UserCoRelLogitWithAggre(nn.Module):
    def __init__(self, hidden_dim, rv_len, cosine_interaction, word_score_layer, rel_score_layer, temperature):
        super().__init__() 

        self.hidden_dim = hidden_dim
        self.rv_len = rv_len
        self.interaction = cosine_interaction
        self.word_score_layer = word_score_layer
        self.rel_score_layer = rel_score_layer
        self.temp = temperature

    def forward(self, seq_a, seq_b, mask_a, mask_b):
        """
        Args: 
            seq_a: [bz, rv_num, rv_len, hdim]
            seq_b: [bz, rv_num, rv_len, hdim]
            mask_a: [bz, rv_num, rv_len]
            mask_b:
        
        Return:
            ab_logits: [bz, rv_num]
            aggre_seq_a: [bz, rv_num ,hdim]
        """
        bz, rv_num, rv_len, hdim = list(seq_a.size())
        assert hdim == self.hidden_dim

        expand_seq_b = seq_b.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        expand_mask_b = mask_b.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, 1, rv_num*rv_len)
        seq_a = seq_a.view(bz*rv_num, rv_len, hdim)
        mask_a = mask_a.view(bz*rv_num, rv_len)
        
        atob_cos_affinity = self.interaction(seq_a, expand_seq_b) #[*, rv_len, rv_num*rv_len]
        ab_mean_feat = masked_colwise_mean(atob_cos_affinity, expand_mask_b) #[*, rv_len, 1]
        ab_max_feat, _ = atob_cos_affinity.max(dim=-1, keepdim=True) #[*, rv_len, 1]
        ab_feat = torch.cat([ab_mean_feat, ab_max_feat], dim=-1).view(bz*rv_num, rv_len, 2)
        ab_word_score = self.word_score_layer(seq_a, mask_a).view(bz*rv_num, rv_len, 1)
        ab_feat = (ab_feat * ab_word_score).view(bz*rv_num, rv_len*2)
        ab_logits = self.rel_score_layer(ab_feat).view(bz, rv_num) # [bz, rv_num]

        ab_word_score = torch.softmax(torch.masked_fill(ab_word_score.view(bz*rv_num, rv_len), 
                            ~mask_a, -1e8) / self.temp, dim=1) #[bz*rv_num, rv_len, 1]
        aggre_seq_a = attention_weighted_sum(ab_word_score, seq_a).view(bz, rv_num, hdim)

        return ab_logits, aggre_seq_a





class SingleRelLogit(nn.Module):
    def __init__(self, hidden_dim, rv_len, cosine_interaction, word_score_layer, rel_score_layer):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cosine_interaction = cosine_interaction
        self.word_score_layer = word_score_layer
        self.rel_score_layer = rel_score_layer

    def forward(self, ui_seq, seq_b, ui_seq_mask, mask_b):
        """
        Args:
            ui_seq: [bz, rv_length, dim]
            seq_b: [bz, rv_num, rv_length, dim]
            ui_mask: [bz, rv_length]
            mask_b: [bz, rv_num, rv_length]
        """
        bz, rv_num, rv_len, hdim = list(seq_b.size())

        seq_b = seq_b.view(bz, rv_num*rv_len, -1) #[bz, rv_numxrv_len, hdim]
        mask_b = mask_b.view(bz, rv_num*rv_len).unsqueeze(1) #[bz, 1, rv_num*rv_len]

        cosine_affinity = self.cosine_interaction(ui_seq, seq_b) #[bz, rv_len, rv_numxrv_len]
        mean_feature = masked_colwise_mean(cosine_affinity, mask_b) #[bz, rv_len, 1]
        max_feature, _ = cosine_affinity.max(dim=-1, keepdim=True) #[bz, rv_len, 1]
        out_feat = torch.cat([mean_feature, max_feature], dim=-1) #[bz, rv_len, 2]
        word_score = self.word_score_layer(ui_seq, ui_seq_mask).view(bz, rv_len, 1) #[bz, rv_len, 1]
        
        out_feat = (out_feat * word_score).view(bz, rv_len*2)
        rel_logit = self.rel_score_layer(out_feat) #[bz, 1]

        return rel_logit

