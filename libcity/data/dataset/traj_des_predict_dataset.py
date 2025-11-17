import torch
from torch.utils.data import DataLoader

from libcity.data.dataset import BaseDataset, padding_mask


class TrajDesPredictDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.predict_length = config.get('predict_length',4)
        self.collate_fn = collate_superv_classify_des
    def _gen_dataloader(self, train_dataset, eval_dataset, test_dataset):
        assert self.collate_fn is not None  # 断言检查（assert condition,message）,如果condition为true 继续执行，否则 输出message（可选）
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=True,
                                      collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                           vocab=self.vocab, add_cls=self.add_cls,
                                                                           predict_length=self.predict_length))
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=True,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab, add_cls=self.add_cls,
                                                                          predict_length=self.predict_length))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False,
                                     collate_fn=lambda x: self.collate_fn(x, max_len=self.seq_len,
                                                                          vocab=self.vocab,
                                                                          add_cls=self.add_cls,
                                                                          predict_length=self.predict_length))  # 指定collate_fn函数，x为函数的默认参数，该默认参数的实值为Dataset.__getitem__()的返回值所组成的列表，长度为batch_size
        return train_dataloader, eval_dataloader, test_dataloader

def collate_superv_classify_des(data, max_len=None, vocab=None, add_cls=False, predict_length=4):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    labels = []
    update_length = []
    for i in range(batch_size):
        if lengths[i] <= max_len:
            end = lengths[i] - predict_length
        else:
            end = max_len - predict_length
        update_length.append(end)
        X[i, :end, :] = features[i][:end, :]
        labels.append(features[i][-predict_length:,0])   # 取每条轨迹的最后若干个点作为标签 (batch_size, predict_length)
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
    targets = torch.stack(labels,0)

    padding_masks = padding_mask(torch.tensor(update_length, dtype=torch.int16), max_len=max_len)

    return X.long(), targets.long(), padding_masks, batch_temporal_mat.long()  # batch_temporal_mat全0
