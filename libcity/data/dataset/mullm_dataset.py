import numpy as np
from libcity.data.dataset import BaseDataset, WordVocab, TrajectoryProcessingDataset, padding_mask
import torch

from libcity.utils.utils import compute_time_deltas


class MulLMDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_unsuperv_two_mask
        self.masking_ratio = self.config.get('masking_ratio', 0.2)
        self.masking_mode = self.config.get('masking_mode', 'together')
        self.distribution = self.config.get('distribution', 'random')
        self.avg_mask_len = self.config.get('avg_mask_len', 3)
        self.avg_len_full_mask = self.config.get('avg_len_full_mask', 3)
        # self.avg_len_time_mask = self.config.get('avg_len_time_mask', 3)
        self.time_masking_ratio = self.config.get('time_masking_ratio', 0.2)

    def _gen_dataset(self):
        train_dataset = MulLMSubDataset(data_name=self.dataset, data_type='train',
                                        vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                        merge=self.merge, min_freq=self.min_freq,
                                        max_train_size=self.max_train_size,
                                        full_masking_ratio=self.masking_ratio,
                                        time_masking_ratio=self.time_masking_ratio,
                                        masking_mode=self.masking_mode, distribution=self.distribution,
                                        avg_mask_len=self.avg_mask_len)
        eval_dataset = MulLMSubDataset(data_name=self.dataset, data_type='eval',
                                       vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                       merge=self.merge, min_freq=self.min_freq,
                                       max_train_size=None,
                                       full_masking_ratio=self.masking_ratio,
                                       time_masking_ratio=self.time_masking_ratio,
                                       masking_mode=self.masking_mode, distribution=self.distribution,
                                       avg_mask_len=self.avg_mask_len)
        test_dataset = MulLMSubDataset(data_name=self.dataset, data_type='test',
                                       vocab=self.vocab, seq_len=self.seq_len, add_cls=self.add_cls,
                                       merge=self.merge, min_freq=self.min_freq,
                                       max_train_size=None,
                                       full_masking_ratio=self.masking_ratio,
                                       time_masking_ratio=self.time_masking_ratio,
                                       masking_mode=self.masking_mode, distribution=self.distribution,
                                       avg_mask_len=self.avg_mask_len)
        return train_dataset, eval_dataset, test_dataset

class MulLMSubDataset(TrajectoryProcessingDataset):
    def __init__(self, data_name, data_type, vocab, seq_len=512, add_cls=True,
                 merge=True, min_freq=1, max_train_size=None, full_masking_ratio=0.2, time_masking_ratio=0.2, masking_mode='together',
                 distribution='random', avg_mask_len=3):
        super().__init__(data_name, data_type, vocab, seq_len, add_cls, merge, min_freq, max_train_size)
        self.max_train_size = max_train_size
        self.full_masking_ratio = full_masking_ratio
        self.time_masking_ratio = time_masking_ratio
        self.masking_mode = masking_mode
        self.distribution = distribution
        self.avg_mask_len = avg_mask_len
        self.exclude_feats = None

    def __getitem__(self, ind):
        traj_ind = self.traj_list[ind]  # (seq_length, feat_dim)
        temporal_mat = self.temporal_mat_list[ind]  # (seq_length, seq_length)
        # todo 实现mulmask：full_mask 之外的 路段 随机选取一部分 mask time
        # todo done
        full_mask, time_mask = two_noise_mask(traj_ind, self.full_masking_ratio,self.time_masking_ratio, self.avg_mask_len,
                                              self.masking_mode, self.distribution, self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array

        return torch.LongTensor(traj_ind), torch.LongTensor(full_mask), torch.LongTensor(time_mask), torch.LongTensor(temporal_mat)



def two_noise_mask(X, masking_ratio, time_masking_ratio, lm=3, mode='together', distribution='random', exclude_feats=None, add_cls=True):
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)
    # full mask
    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            full_mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    full_mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            full_mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    elif distribution == 'random':  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            full_mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            full_mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])
    else:
        full_mask = np.ones(X.shape, dtype=bool)
    # time mask using random
    time_mask = np.ones(X.shape, dtype=bool)
    unmasked_indices = np.where(full_mask[:, 0])[0] # 未被mask掉的样本索引
    time_mask_in_pos = np.random.choice(np.array([True,False]), size = unmasked_indices.shape[0], replace=True,
                                     p=(1-time_masking_ratio, time_masking_ratio))

    time_mask[unmasked_indices, :] = np.tile(time_mask_in_pos[:, None], (1, X.shape[1]))
    # time_mask[unmasked_indices] = time_mask_in_pos
    # time_mask = np.tile(time_mask[:,None], X.shape[1])
    time_mask[0] = True # 时间维度不预测第一个
    if add_cls:
        full_mask[0] = True  # CLS at 0, set mask=1
        time_mask[0] = True
        time_mask[1] = True
    return full_mask, time_mask


def collate_unsuperv_two_mask(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, full_masks, time_masks, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    # masks related to objective
    target_full_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    target_time_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_full_masks[i, :end, :] = full_masks[i][:end, :]
        target_time_masks[i, :end, :] = time_masks[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len) #  False: 填充的位置

    target_full_masks = ~target_full_masks  # (batch_size, padded_length, feat_dim) 取反之后，True:被mask掉用于预测的位置
    target_time_masks = ~target_time_masks
    target_full_masks = target_full_masks * padding_masks.unsqueeze(-1)
    target_time_masks = target_time_masks * padding_masks.unsqueeze(-1)
    # 目标mask与padmask结合，保证预测的位置不是填充的 True：用于预测的位置

    full_targets = X.clone()
    full_targets = full_targets.masked_fill_(target_full_masks == 0, vocab.pad_index)
    # time_targets = X.clone()
    time_targets = compute_time_deltas(X)

    time_targets = time_targets.masked_fill_(target_time_masks[...,1] == 0, vocab.pad_index)


    X[..., 0:1].masked_fill_(target_full_masks[..., 0:1] == 1, vocab.mask_index)  # loc -> mask_index
    X[..., 1:].masked_fill_(target_full_masks[..., 1:] == 1, vocab.pad_index)  # others -> pad_index
    X[..., 1:4].masked_fill_(target_time_masks[..., 1:4] == 1, vocab.mask_index)  # time -> mask_index
    # 处理时间矩阵
    # 要把mask掉的预测位置所在的整行和整列都设为0
    temporal_mat_masks = target_full_masks[...,0].unsqueeze(2) | target_time_masks[...,0].unsqueeze(1)
    batch_temporal_mat.masked_fill_(temporal_mat_masks.bool(), vocab.mask_index)
    return X.long(), full_targets.long(), target_full_masks, time_targets.float(),target_time_masks, \
        padding_masks, batch_temporal_mat.long()


def geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask
