from logging import getLogger

import math
import inspect
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from libcity.model.layers.nodeembedding import NodeEmbedding2, MultiBinsIntervalEmbedding, NodeEmbedding4
from libcity.model.trajectory_embedding.BERT import Mlp, DropPath


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class AttnMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = self.config.get("d_model", 768)
        self.dropout = self.config.get("dropout", 0.2)
        self.bias = False
        self.n_linear = 1
        self.lin_layers = nn.ModuleList(
            [nn.Linear(self.n_embd, self.n_embd, bias=True) for _ in range(self.n_linear)])
        self.dropout = nn.Dropout(self.dropout)
        self.ln_layers = nn.ModuleList([LayerNorm(self.n_embd, bias=self.bias) for _ in range(self.n_linear)])

    def forward(self, x):
        for linear, ln in zip(self.lin_layers, self.ln_layers):
            x = new_gelu(linear(ln(x)))
        return x


class AttnMLP2(nn.Module):
    def __init__(self, d_model=768, dropout=0.2, bias=False, n_linear=1):
        super().__init__()
        self.n_embd = d_model
        self.dropout = dropout
        self.bias = bias
        self.n_linear = n_linear
        self.lin_layers = nn.ModuleList(
            [nn.Linear(self.n_embd, self.n_embd, bias=True) for _ in range(self.n_linear)])
        self.dropout = nn.Dropout(self.dropout)
        self.ln_layers = nn.ModuleList([LayerNorm(self.n_embd, bias=self.bias) for _ in range(self.n_linear)])

    def forward(self, x):
        for linear, ln in zip(self.lin_layers, self.ln_layers):
            x = new_gelu(linear(ln(x)))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = self.config.get("d_model", 768)
        self.n_head = self.config.get('attn_heads', 12)
        self.dropout = self.config.get("dropout", 0.2)
        # self.block_size = self.config.get("block_size", 24)  ERROR!
        self.block_size = self.config.get("seq_len", 128)
        self.bias = False
        assert self.n_embd % self.n_head == 0
        # proj function
        self.value_mlp = AttnMLP(config)
        self.query_and_key_mlp = AttnMLP(config)
        # Q, K, V transformation
        self.query_transform = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.key_transform = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.value_transform = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.n_head = self.n_head
        self.n_embd = self.n_embd
        self.dropout = self.dropout
        # support only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("causal_mask", torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1,
                                                                                                              self.block_size,
                                                                                                              self.block_size))

    def forward(self, x, padding_masks=None):
        B, T, C = x.size()
        unshare_x = self.value_mlp(x)
        share_x = self.query_and_key_mlp(x)
        q = self.query_transform(share_x)
        k = self.key_transform(share_x)
        v = self.value_transform(unshare_x)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:  # todo 由于 scaled_dot_product_attention 暂时没有支持padding_masks,需要手工构造attn_mask
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout,
                                                                 is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            # padding_masks
            if padding_masks is not None:
                # padding_masks: (B, T) —> expand to (B, 1, 1, T)
                padding_mask_expanded = padding_masks[:, None, None, :].to(dtype=torch.bool)
                att = att.masked_fill(~padding_mask_expanded, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, d_model, dim_out, attn_drop=0., proj_drop=0.,
                 add_cls=True, device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        super().__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.device = device
        self.add_cls = add_cls
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval

        self.query_and_key_linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])
        self.value_transform = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=attn_drop)
        # private for value and public for query and key
        self.value_mlp = AttnMLP2(d_model, proj_drop, False, 1)
        self.query_and_key_mlp = AttnMLP2(d_model, proj_drop, False, 1)

        self.c_proj = nn.Linear(d_model, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = nn.Linear(1, self.temporal_bias_dim, bias=True)
                self.temporal_mat_bias_2 = nn.Linear(self.temporal_bias_dim, 1, bias=True)
            elif self.temporal_bias_dim == -1:
                self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
                nn.init.xavier_uniform_(self.temporal_mat_bias)

    def forward(self, x, padding_masks, future_mask=False, output_attentions=False, batch_temporal_mat=None):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T) padding_mask
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:

        """
        batch_size, seq_len, d_model = x.shape
        # 0) private linear projection and public linear projection
        private_x = self.value_mlp(x)
        share_x = self.query_and_key_mlp(x)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # l(x) --> (B, T, d_model)
        # l(x).view() --> (B, T, head, d_k)
        query, key = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                      for l, x in zip(self.query_and_key_linear_layers, (share_x, share_x))]
        value = self.value_transform(private_x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # q, k, v --> (B, head, T, d_k)

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, head, T, T)

        if self.add_temporal_bias:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) +
                    (batch_temporal_mat / torch.tensor(60.0).to(self.device)))
            else:
                batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + batch_temporal_mat)
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(F.leaky_relu(
                    self.temporal_mat_bias_1(batch_temporal_mat.unsqueeze(-1)),
                    negative_slope=0.2)).squeeze(-1)  # (B, T, T)
            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias.expand((1, seq_len, seq_len))
            batch_temporal_mat = batch_temporal_mat.unsqueeze(1)  # (B, 1, T, T)
            scores += batch_temporal_mat  # (B, 1, T, T)

        if padding_masks is not None:
            scores.masked_fill_(padding_masks == 0, float('-inf'))

        if future_mask:
            mask_postion = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).bool().to(self.device)
            if self.add_cls:
                mask_postion[:, 0, :] = 0
            scores.masked_fill_(mask_postion, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)  # (B, head, T, T)
        p_attn = self.dropout(p_attn)
        out = torch.matmul(p_attn, value)  # (B, head, T, d_k)

        # 3) "Concat" using a view and apply a final linear.
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # (B, T, d_model)
        out = self.c_proj(out)  # (B, T, N, D)
        out = self.proj_drop(out)
        if output_attentions:
            return out, p_attn  # (B, T, dim_out), (B, head, T, T)
        else:
            return out, None  # (B, T, dim_out)


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, drop_path,
                 attn_drop, dropout, type_ln='pre', add_cls=True,
                 device=torch.device('cpu'), add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False):
        """

        Args:
            d_model: hidden size of transformer
            attn_heads: head sizes of multi-head attention
            feed_forward_hidden: feed_forward_hidden, usually 4*d_model
            drop_path: encoder dropout rate
            attn_drop: attn dropout rate
            dropout: dropout rate
            type_ln:
        """

        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, d_model=d_model, dim_out=d_model,
                                              attn_drop=attn_drop, proj_drop=dropout, add_cls=add_cls,
                                              device=device, add_temporal_bias=add_temporal_bias,
                                              temporal_bias_dim=temporal_bias_dim,
                                              use_mins_interval=use_mins_interval)
        self.mlp = Mlp(in_features=d_model, hidden_features=feed_forward_hidden,
                       out_features=d_model, act_layer=nn.GELU, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.type_ln = type_ln

    def forward(self, x, padding_masks, future_mask=True, output_attentions=False, batch_temporal_mat=None):
        """

        Args:
            x: (B, T, d_model)
            padding_masks: (B, 1, T, T)
            future_mask: True/False
            batch_temporal_mat: (B, T, T)

        Returns:
            (B, T, d_model)

        """
        if self.type_ln == 'pre':
            attn_out, attn_score = self.attention(self.norm1(x), padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            attn_out, attn_score = self.attention(x, padding_masks=padding_masks,
                                                  future_mask=future_mask, output_attentions=output_attentions,
                                                  batch_temporal_mat=batch_temporal_mat)
            x = self.norm1(x + self.drop_path(attn_out))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            raise ValueError('Error type_ln {}'.format(self.type_ln))
        return x, attn_score


class MetaMulLMLearning(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.config = config
        self.vocab_size = data_feature.get("vocab_size")
        self.token_size = self.vocab_size  # 这俩相等
        self.d_model = self.config.get("d_model", 768)
        self.seq_len = self.config.get("seq_len", 128)
        self.model = MetaTRL(config, data_feature)
        self.lm_head = nn.Linear(self.d_model, self.token_size, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.reg_head = nn.Linear(self.d_model, 1, bias=True)

    def forward(self, x, padding_masks=None, temporal_mat=None, graph_dict=None):
        emb = self.model(x, padding_masks, temporal_mat, graph_dict)
        logits = self.softmax(self.lm_head(emb))
        predict_t = self.reg_head(emb).squeeze(-1)
        return logits, predict_t




class MetaTRL(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.config = config
        self.device = self.config.get('device', torch.device('cpu'))
        self.vocab_size = data_feature.get("vocab_size")
        self.token_size = self.vocab_size
        self.node_fea_dim = data_feature.get('node_fea_dim')
        self.add_cls = self.config.get('add_cls', False)

        # self.use_start_letter = self.config.get("use_start_letter", True)
        self.d_model = self.config.get("d_model", 768)
        self.dropout = self.config.get("dropout", 0.2)
        self.attn_drop = self.config.get('attn_drop', 0.1)
        self.n_layers = self.config.get("n_layers", 6)
        self.attn_heads = self.config.get('attn_heads', 12)
        self.mlp_ratio = self.config.get('mlp_ratio', 4)
        self.feed_forward_hidden = self.d_model * self.mlp_ratio
        self.type_ln = self.config.get('type_ln', 'pre')
        self.drop_path = self.config.get('drop_path', 0.3)
        self.domain_specific_params = self.config.get("domain_specific_params", ['value_mlp', 'value_transform', 'wte'])
        self.domain_shared_params = self.config.get("domain_shared_params",['meta_features_mlp'])

        self.add_gat = self.config.get('add_gat', False)
        self.add_meta_gat = self.config.get('add_meta_gat', False)
        print("metagat?")
        print(self.add_meta_gat)
        self.add_time_in_day = self.config.get('add_time_in_day', True)
        self.add_day_in_week = self.config.get('add_day_in_week', True)
        self.add_time_interval = self.config.get('add_time_interval', False)
        self.add_share_time_interval = self.config.get('add_share_time_interval', True)
        self.max_time_scale_s = self.config.get('max_time_scale_s', 3000)
        self.time_interval_scale_list = self.config.get('time_interval_scales', [10])
        self.gat_heads_per_layer = self.config.get('gat_heads_per_layer', [8, 1])
        self.gat_features_per_layer = self.config.get('gat_features_per_layer', [16, self.d_model])
        self.gat_dropout = self.config.get('gat_dropout', 0.6)
        self.gat_avg_last = self.config.get('gat_avg_last', True)
        self.load_trans_prob = self.config.get('load_trans_prob', False)
        self.future_mask = self.config.get('future_mask', False)

        self.add_temporal_bias = self.config.get('add_temporal_bias', False)
        self.temporal_bias_dim = self.config.get('temporal_bias_dim', 64)
        self.use_mins_interval = self.config.get('use_mins_interval', False)

        assert self.vocab_size is not None
        self.wte = NodeEmbedding4(d_model=self.d_model, dropout=self.dropout,
                                  add_time_in_day=self.add_time_in_day, add_day_in_week=self.add_day_in_week,
                                  add_time_interval=self.add_time_interval,
                                  max_time_scale_s=self.max_time_scale_s,
                                  time_interval_scale_list=self.time_interval_scale_list,
                                  add_pe=True, node_fea_dim=self.node_fea_dim, add_gat=self.add_gat,
                                  add_meta_gat=self.add_meta_gat, add_hgnn=False,
                                  gat_heads_per_layer=self.gat_heads_per_layer,
                                  gat_features_per_layer=self.gat_features_per_layer,
                                  gat_dropout=self.gat_dropout,
                                  load_trans_prob=self.load_trans_prob, avg_last=self.gat_avg_last)
        if self.add_share_time_interval:
            self.sharedIntervalEmbedding = MultiBinsIntervalEmbedding(num_bins=100, d_model=self.d_model)
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_layers)]  # stochastic depth decay rule
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, attn_heads=self.attn_heads,
                              feed_forward_hidden=self.feed_forward_hidden, drop_path=enc_dpr[i],
                              attn_drop=self.attn_drop, dropout=self.dropout,
                              type_ln=self.type_ln, add_cls=self.add_cls,
                              device=self.device, add_temporal_bias=self.add_temporal_bias,
                              temporal_bias_dim=self.temporal_bias_dim,
                              use_mins_interval=self.use_mins_interval) for i in range(self.n_layers)])

        # self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=True)
        # self.softmax = nn.LogSoftmax(dim=-1)
        # self.transformer.wte.weight = self.lm_head.weight

        # !!! ATTENTION !!!
        # duplicate parameters will not be included in 'named_parameters', i.e., the parameters of lm_head are private as same as wte and not involved in meta updating.
        # if changing the order of initialization, SPEC_DICT['sharemlp'] should add 'lm_head' to ensure the wte parameters (copy from lm_head) are not shared by all cities.
        self.apply(self._init_weights)  # 所有子模块初始化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # def copy_invariant_params(self, city_model):
    #     for (m_name, m_param), (c_name, c_param) in zip(self.named_parameters(), city_model.named_parameters()):
    #         # !!! ATTENTION !!!  对共享参数进行复制
    #         contains_specific = any(sub_str in m_name for sub_str in self.domain_specific_params)
    #         if not contains_specific:
    #             assert m_name == c_name
    #             c_param.data = m_param.data.clone()
    #             assert torch.allclose(c_param.data, m_param.data)

    def copy_invariant_params(self, city_model):
        for (m_name, m_param), (c_name, c_param) in zip(self.named_parameters(), city_model.named_parameters()):
            # 优先判断是否出现在强制共享列表中
            contains_force_shared = any(sub_str in m_name for sub_str in self.domain_shared_params)

            # 原有的 domain-specific 判断
            contains_specific = any(sub_str in m_name for sub_str in self.domain_specific_params)

            # 只要在共享列表中，就一定复制
            if contains_force_shared or not contains_specific:
                assert m_name == c_name
                c_param.data = m_param.data.clone()
                assert torch.allclose(c_param.data, m_param.data)

    def forward(self, x, padding_masks=None, temporal_mat=None, graph_dict=None):
        # device = x.device
        b, t, _ = x.size()
        embedding_output = self.wte(sequence=x, batch_temporal_mat_list=temporal_mat, graph_dict=graph_dict)
        if self.add_share_time_interval:
            inteval_embedding = self.sharedIntervalEmbedding(temporal_mat)  # (B, T, d)
            embedding_output = embedding_output + inteval_embedding
        # pos_emb = self.transformer.wpe(pos)
        padding_masks_input = padding_masks.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, T, T)
        for transformer in self.transformer_blocks:
            embedding_output, attn_score = transformer.forward(
                x=embedding_output, padding_masks=padding_masks_input,
                future_mask=self.future_mask, batch_temporal_mat=temporal_mat)  # (B, T, d_model)
        return embedding_output

    # def crop_block_size(self, seq_length):
    #     assert seq_length <= self.config.seq_length
    #     self.config.seq_length = seq_length
    #     self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:seq_length])
    #     for block in self.transformer.h:
    #         block.attn.bias = block.attn.bias[:, :, :seq_length, :seq_length]

    # def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear,)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = '%s.%s' % (mn, pn) if mn else pn
    #             if pn.endswith('bias'):
    #                 no_decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
    #                 decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
    #                 no_decay.add(fpn)
    #
    #     decay.remove('lm_head.weight')
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     inter_params = decay & no_decay
    #     union_params = decay | no_decay
    #     assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    #     assert len(
    #         param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
    #                                                 % (str(param_dict.keys() - union_params),)
    #     optim_groups = [
    #         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
    #         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    #     ]
    #     use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    #     print(f"using fused AdamW: {use_fused}")
    #     extra_args = dict(fused=True) if use_fused else dict()
    #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    #     return optimizer

    # def estimate_mfu(self, fwdbwd_per_iter, dt):
    #     N = self.get_num_params()
    #     cfg = self.config
    #     L, H, Q, T = cfg.n_layers, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    #     flops_per_token = 6 * N + 12 * L * H * Q * T
    #     flops_per_fwdbwd = flops_per_token * T
    #     flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    #     flops_achieved = flops_per_iter * (1.0 / dt)
    #     flops_promised = 312e12
    #     mfu = flops_achieved / flops_promised
    #     return mfu

    @torch.no_grad()
    def generate_for_seir(self, args, num_samples, temperature=1.0, top_k=None):
        args.freqs = args.freqs / args.freqs.sum()
        adjustments = np.log(args.freqs ** args.balance_coef + 1e-12)
        adjustments = torch.from_numpy(adjustments)
        adjustments = adjustments.to(args.device)
        start_dist = torch.tensor(np.load(f'{self.config.datapath}/{self.config.data}/start.npy')).float()
        idx = torch.LongTensor([torch.multinomial(start_dist, 1) for _ in range(num_samples)]).reshape(-1, 1).to(
            args.device)
        gen_seq_len = 24 * 7 - 1

        for t in tqdm(range(gen_seq_len)):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # adjust logits when generating
            logits = logits - adjustments
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        pred = []
        for i in range(len(idx)):
            seq = idx[i][:]
            pred.append(list(seq.cpu().numpy()))
        return pred


class MetaTRLDownStream(nn.Module):

    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.pooling = self.config.get('pooling', 'mean')
        self.d_model = self.config.get('d_model', 768)
        self.add_cls = self.config.get('add_cls', True)
        self.baseline_bert = self.config.get('baseline_bert', False)
        self.baseline_tf = self.config.get('baseline_tf', False)

        self._logger = getLogger()
        self._logger.info("Building GAT2COLADownstream model")

        self.cola = MetaTRL(config, data_feature)

    def forward(self, x, padding_masks, batch_temporal_mat, graph_dict):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, feat_dim)
        """
        if self.pooling in ['avg_first_last', 'avg_top2']:
            output_hidden_states = True
        token_emb = self.cola(x, padding_masks, batch_temporal_mat, graph_dict)  # (batch_size, seq_length, d_model)
        if self.pooling == 'cls' or self.pooling == 'cls_before_pooler':
            if self.add_cls:
                return token_emb[:, 0, :]  # (batch_size, feat_dim)
            else:
                raise ValueError('No use cls!')
        elif self.pooling == 'mean':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask  # (batch_size, feat_dim)
        elif self.pooling == 'max':
            input_mask_expanded = padding_masks.unsqueeze(-1).expand(
                token_emb.size()).float()  # (batch_size, seq_length, d_model)
            token_emb[input_mask_expanded == 0] = float('-inf')  # Set padding tokens to large negative value
            max_over_time = torch.max(token_emb, 1)[0]
            return max_over_time  # (batch_size, feat_dim)
        else:
            raise ValueError('Error pooling type {}'.format(self.pooling))


class MetaTRLLinearETA(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.vocab_size = data_feature.get('vocab_size')
        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearETA model")
        self.colads = MetaTRLDownStream(config, data_feature)
        # self.colads = COLADownstream(config, data_feature)
        # self.pooler = SequenceMeanPoolLayer(self.d_model, self.d_model)
        self.linear = nn.Linear(self.d_model, 1)

    def forward(self, x, padding_masks=None, batch_temporal_mat=None, graph_dict=None):
        # traj_emb, _, _ = self.model(x=x, graph_dict=graph_dict)  # (B, patch_num, d_model)
        # traj_emb = self.colads(x, padding_masks)  # (B, patch_num, d_model)
        traj_emb = self.colads(x, padding_masks, batch_temporal_mat, graph_dict)  # (B, patch_num, d_model)
        # traj_emb = self.pooler(traj_emb)  # (B, d_model)
        eta_pred = self.linear(traj_emb)  # (B, 1)
        return eta_pred  # (B, 1)


class MetaTRLLinearClassify(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()

        self.config = config

        self.usr_num = data_feature.get('usr_num')
        self.d_model = self.config.get('d_model', 768)
        self.dataset = self.config.get('dataset', '')
        self.classify_label = self.config.get('classify_label', 'vflag')

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearClassify model")

        # self.model = GAT2PatchTST(config, data_feature)
        self.colads = MetaTRLDownStream(config, data_feature)
        # self.colads = COLADownstream(config,data_feature)
        # self.pooler = SequenceMeanPoolLayer(self.d_model, self.d_model)
        if self.classify_label == 'vflag':
            self.linear = nn.Linear(self.d_model, 2)
            if self.dataset == 'geolife':
                self.linear = nn.Linear(self.d_model, 4)
        elif self.classify_label == 'usrid':
            self.linear = nn.Linear(self.d_model, self.usr_num)
        else:
            raise ValueError('Error classify_label = {}'.format(self.classify_label))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, padding_masks=None, batch_temporal_mat=None, graph_dict=None):
        # traj_emb, _, _ = self.model(x=x, graph_dict=graph_dict)  # (B, d_model)
        traj_emb = self.colads(x, padding_masks, batch_temporal_mat, graph_dict)  # (B, d_model)
        # traj_emb = self.colads(x, padding_masks)  # (B, d_model)
        # traj_emb = self.pooler(traj_emb)
        nloc_pred = self.softmax(self.linear(traj_emb))  # (B, n_class)  加了一层全连接层
        return nloc_pred  # (B, n_class)




class MetaTRLLinearDesPredict(nn.Module):
    def __init__(self, config, data_feature):
        super().__init__()
        self.config = config
        self.usr_num = data_feature.get('usr_num')
        self.vocab_size = data_feature.get('vocab_size')

        self.d_model = self.config.get('d_model', 768)
        self.dataset = self.config.get('dataset', '')
        self.predict_length = self.config.get('predict_length', 4)  # 预测长度

        self._logger = getLogger()
        self._logger.info("Building Downstream LinearDesPredict model")

        # self.model = GAT2PatchTST(config, data_feature)
        self.colads = MetaTRLDownStream(config, data_feature)
        # self.colads = COLADownstream(config,data_feature)
        # self.pooler = SequenceMeanPoolLayer(self.d_model, self.d_model)
        self.linear = nn.Linear(self.d_model, self.predict_length * self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, padding_masks=None, batch_temporal_mat=None, graph_dict=None):
        traj_emb = self.colads(x, padding_masks, batch_temporal_mat, graph_dict)
        dest_pred = self.softmax(self.linear(traj_emb).view(-1, self.predict_length, self.vocab_size))
        return dest_pred
