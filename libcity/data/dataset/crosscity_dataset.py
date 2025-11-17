import torch

from libcity.data.dataset import BaseDataset, padding_mask


class CrossCityDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_superv_nextpos


def collate_superv_nextpos(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)

    Y = torch.zeros(batch_size, max_len,dtype=torch.long)
    Y_lengths = [max(length - 1, 0) for length in lengths]  # Y 的有效长度
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
    Y[:, :-1] = X[:, 1:, 0]

    # 两种情况分析，最后构造的Y是一致的
    # if add_cls:  # 要确保轨迹的最后一个点不参与损失计算，因为它没有下一个位置
    #     # X的第一个位置是cls
    #     for i in range(batch_size):
    #         end = min(lengths[i], max_len)
    #         X[i,:end,:] = features[i][:end,:]
    #         batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
    #     Y[:,:] = X[:,1:,0]
    #         # Y[i,:] = X[i,:-1,0]
    # else:
    #     for i in range(batch_size):
    #         end = min(lengths[i], max_len)
    #         X[i,:end,:] = features[i][:end,:]
    #         batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
    #     Y[:,:] = X[:,1:,0]
    # for i in range(batch_size):
    #     end = min(lengths[i], max_len)
    #     X[i, :end, :] = features[i][:end, :]
    #     labels.append(features[i][-1][5])
    #     batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    targets_masks = padding_mask(torch.tensor(Y_lengths, dtype=torch.int16), max_len=max_len)
    Y[~targets_masks] = -1
    # targets = torch.LongTensor(Y)  # (batch_size,seq_len)

    return X.long(), Y.long(), padding_masks, targets_masks, batch_temporal_mat.long()
