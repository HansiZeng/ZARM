import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .layers import WordEmbedding, NgramFeatWithSeq, AddictiveAttention, LastFeat, FM, VariationalDropout
from .layers import UserCoRelLogit, CosineInteraction, WordScore, SingleRelLogit, ZeroAttention
from .utils import get_rev_mask, attention_weighted_sum


class RelevanceHsanHighway(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, kernel_size, latent_dim,
                vocab_size, user_size, item_size, rv_len,
                pretrained_embeddings, dropout, sparse, word_dropout, mode):
        super().__init__() 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.seq_dim = self.hidden_dim

        # highway network 
        self.word_embedding = WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings,
                                            padding_idx=0, sparse=sparse)
        self.var_dropout = VariationalDropout(p=word_dropout)
        
        self.ngram_feat_word_layer = NgramFeatWithSeq(embedding_dim, hidden_dim, kernel_size, num_head=2, dropout=0.3)
        self.ngram_feat_sent_layer = NgramFeatWithSeq(hidden_dim, hidden_dim, kernel_size, num_head=2, dropout=0.3)

        self.review_att_layer = AddictiveAttention(hidden_dim, latent_dim)

        # relevance network 
        self.smt_to_rel_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.seq_dim),
                                                nn.Tanh())
        if "user" in self.mode or "item" in self.mode:
            self._cosine_interaction = CosineInteraction() 
            self._word_score_layer = WordScore(self.seq_dim)
            self._rel_score_layer = nn.Sequential(nn.Dropout(p=0.1),
                                                nn.Linear(rv_len*2, 16),
                                                nn.Tanh(),
                                                nn.Linear(16, 1))
            self.user_rel_logit_layer = UserCoRelLogit(self.seq_dim, rv_len, 
                        cosine_interaction=self._cosine_interaction, word_score_layer=self._word_score_layer, 
                        rel_score_layer=self._rel_score_layer)
            self.single_rel_logit_layer = SingleRelLogit(self.seq_dim, rv_len, 
                    cosine_interaction=self._cosine_interaction, word_score_layer=self._word_score_layer, 
                    rel_score_layer=self._rel_score_layer)
            self.zero_attention_layer = ZeroAttention()
            self.proj_layer = nn.Linear(self.seq_dim+self.hidden_dim, self.hidden_dim)

            self.user_last_feat_layer = LastFeat(user_size,  hidden_dim, latent_dim, padding_idx=0)
            self.item_last_feat_layer = LastFeat(item_size, hidden_dim, latent_dim, padding_idx=0)
        else:
            self.user_last_feat_layer = LastFeat(user_size, hidden_dim, latent_dim, padding_idx=0)
            self.item_last_feat_layer = LastFeat(item_size, hidden_dim, latent_dim, padding_idx=0)

        self.fm = FM(user_size, item_size, latent_dim, dropout, user_padding_idx=0, item_padding_idx=0)

    def forward(self, u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_sent_masks, i_rev_sent_masks,
                u_rev_masks, i_rev_masks, u_ids, i_ids,
                ui_rev, neg_ui_rev, ui_mask, neg_ui_mask, training):
        """
        Args:
            u_revs: [bz, rv_num, sent_num, word_num]
            i_revs: [bz, rv_num, sent_num, word_num]
            u_rev_word_masks: [bz, rv_num, sent_num, word_num]
            i_rev_word_masks: [bz, rv_num, sent_num, word_num]
            u_rev_sent_masks: [bz, rv_num, sent_num]
            i_rev_sent_masks: [bz, rv_num, sent_num]
            u_rev_masks: [bz, rv_num]
            i_rev_masks: [bz, rv_num]
            u_ids: [bz]
            i_ids: [bz]
            ui_rev: [bz, sent_num, word_num]
            neg_ui_rev: [bz, sent_num, word_num]
        
        Returns:
            out_logits: [bz]
            u_rev_scores: [bz, rv_num]
            i_rev_scores: [bz, ]
        """
        bz, rv_num, sent_num, word_num = list(u_revs.size())
        rv_len = sent_num * word_num

        # each review representation
        u_rev_seqs = self.var_dropout(self.word_embedding(u_revs).view(bz*rv_num*sent_num, word_num, self.embedding_dim))
        u_revs = self.ngram_feat_word_layer(u_rev_seqs, u_rev_word_masks.view(bz*rv_num*sent_num, word_num))[0].view(bz*rv_num, sent_num, self.seq_dim)
        u_revs = self.ngram_feat_sent_layer(u_revs, u_rev_sent_masks.view(bz*rv_num, sent_num))[0].view(bz, rv_num, self.seq_dim)
        u_rev_seqs = u_rev_seqs.view(bz*rv_num, rv_len, self.embedding_dim)

        i_rev_seqs = self.var_dropout(self.word_embedding(i_revs).view(bz*rv_num*sent_num, word_num, self.embedding_dim))
        i_revs = self.ngram_feat_word_layer(i_rev_seqs, i_rev_word_masks.view(bz*rv_num*sent_num, word_num))[0].view(bz*rv_num, sent_num, self.seq_dim)
        i_revs = self.ngram_feat_sent_layer(i_revs, i_rev_sent_masks.view(bz*rv_num, sent_num))[0].view(bz, rv_num, self.seq_dim)
        i_rev_seqs = i_rev_seqs.view(bz*rv_num, rv_len, self.embedding_dim)

        # user/item representation from highway 
        u_rev_feat, _ = self.review_att_layer(u_revs, u_rev_masks)
        i_rev_feat, _ = self.review_att_layer(i_revs, i_rev_masks)

        # user/item representation from relerance match
        #u_rev_seqs = self.smt_to_rel_layer(u_rev_seqs.view(bz, rv_num, rv_len, self.seq_dim).detach())
        #i_rev_seqs = self.smt_to_rel_layer(i_rev_seqs.view(bz, rv_num, rv_len, self.seq_dim).detach())
        u_rev_seqs = self.smt_to_rel_layer(u_rev_seqs.view(bz, rv_num, rv_len, self.embedding_dim))
        i_rev_seqs = self.smt_to_rel_layer(i_rev_seqs.view(bz, rv_num, rv_len, self.embedding_dim))
        # for user 
        if "user" in self.mode:
            if training:
                #ui_rev = self.smt_to_rel_layer(self.ngram_feat_layer(self.var_dropout(self.word_embedding(ui_rev)), ui_mask)[1].detach())
                #neg_ui_rev = self.smt_to_rel_layer(self.ngram_feat_layer(self.var_dropout(self.word_embedding(neg_ui_rev)), 
                                        #neg_ui_mask)[1].detach())
                ui_rev = ui_rev.view(bz, rv_len)
                ui_mask = ui_mask.view(bz, rv_len)
                neg_ui_rev = neg_ui_rev.view(bz, rv_len)
                neg_ui_mask = neg_ui_mask.view(bz, rv_len)
                u_rev_word_masks = u_rev_word_masks.view(bz, rv_num, rv_len)
                i_rev_word_masks = i_rev_word_masks.view(bz, rv_num, rv_len)
                #print("ui_rev", i_rev_seqs.shape, i_rev_word_masks.shape, ui_rev.shape, ui_mask.shape)
                ui_rev = self.smt_to_rel_layer(self.var_dropout(self.word_embedding(ui_rev)))
                neg_ui_rev = self.smt_to_rel_layer(self.var_dropout(self.word_embedding(neg_ui_rev)))
                ui_logits = self.single_rel_logit_layer(ui_rev, i_rev_seqs, ui_mask, i_rev_word_masks)
                neg_ui_logits = self.single_rel_logit_layer(neg_ui_rev, i_rev_seqs, neg_ui_mask, i_rev_word_masks)

            u_rel_logits = self.user_rel_logit_layer(u_rev_seqs, i_rev_seqs, u_rev_word_masks, i_rev_word_masks) #[bz, rv_num]
            u_rev_seqs = u_rev_seqs.view(bz*rv_num, rv_len, self.seq_dim)
            u_rel_revs = F.max_pool1d(u_rev_seqs.transpose(1,2), kernel_size=rv_len).view(bz, rv_num, self.seq_dim)
            u_rel_feat, u_rel_scores  = self.zero_attention_layer(u_rel_revs, u_rev_masks, u_rel_logits) #[bz, seq_dim]
            u_rev_feat = self.proj_layer(torch.cat([u_rev_feat, u_rel_feat], dim=1))
            # stat
            rel_feat_norm = torch.norm(u_rel_feat, dim=1)
            zero_perc = u_rel_scores[:, -1] / u_rel_scores.sum(dim=1) #[bz]
        
        if "item" in self.mode:
            i_rel_logits = self.user_rel_logit_layer(i_rev_seqs, u_rev_seqs, i_rev_word_masks, u_rev_word_masks)
            i_rev_seqs = i_rev_seqs.view(bz*rv_num, rv_len, self.seq_dim)
            i_rel_revs = F.max_pool1d(i_rev_seqs.transpose(1,2), kernel_size=rv_len).view(bz,rv_num, self.seq_dim)
            i_rel_feat, i_rel_scores = self.zero_attention_layer(i_rel_revs, i_rev_masks, i_rel_logits)
            i_rev_feat = self.proj_layer(torch.cat([i_rev_feat, i_rel_feat], dim=1))


        # user/item combine representation 
        u_feat = self.user_last_feat_layer(u_rev_feat, u_ids)
        i_feat = self.item_last_feat_layer(i_rev_feat, i_ids) #[bz, hdim]

        # fm 
        out_logits = self.fm(u_feat, i_feat, u_ids, i_ids)

        if training:
            return out_logits.view(bz), (ui_logits.view(bz), neg_ui_logits.view(bz)), (rel_feat_norm, zero_perc)
        else:
            return out_logits.view(bz), (rel_feat_norm, zero_perc)

