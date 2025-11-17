import os
import time
import numpy as np
import torch

from libcity.executor import MetaTRLExecutor
import matplotlib as mpl

from libcity.executor.abstract_multi_source_executor import AbstractMultiSourceExecutor
from libcity.utils.utils import cul_batch_time_interval

mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


class MulLMExecutor(MetaTRLExecutor):
    # model:dict {name,model}
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.criterion_cla = torch.nn.NLLLoss(ignore_index=0, reduction='none')
        self.criterion_reg = torch.nn.MSELoss(reduction='none')
        self.cla_loss_weight = self.config.get("cla_loss_weight", 0.5)
        self.reg_loss_weight = self.config.get("reg_loss_weight", 0.5)
        self.initial_ckpt = self.config.get("initial_ckpt", None)
        self.unload_param = self.config.get("unload_param", [])

        self.time_interval_scale = self.config.get("time_interval_scale", 300)
        self.max_interval = self.config.get("max_time_scale_s", 5000)
        self.time_interval_scale_list = self.config.get("time_interval_scales", [])

        self.target_city = self.config.get("dataset", None)
        # 也许需要使用来减小显存
        # ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}["float16"]
        # self.ctx = torch.autocast(device_type='cuda', dtype=ptdtype)

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        start_time = time.time()
        self._valid_city_epoch(test_dataloader, self.target_city, 0, mode='Test')
        t1 = time.time()
        self._logger.info('Test time {}s.'.format(t1 - start_time))

    def _cal_cla_loss(self, pred, targets, targets_mask):
        batch_loss_list = self.criterion_cla(pred.transpose(1, 2), targets)
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
        return mean_loss, batch_loss, num_active

    def _cal_cla_acc(self, pred, targets, targets_mask):
        mask_label = targets[targets_mask]  # (num_active, )
        lm_output = pred[targets_mask].argmax(dim=-1)  # (num_active, )
        correct_l = mask_label.eq(lm_output).sum().item()
        return correct_l

    def _cal_reg_loss(self, pred, targets, targets_mask):  # todo 实现
        batch_loss_list = self.criterion_reg(pred, targets)
        batch_loss_list = batch_loss_list * targets_mask
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = batch_loss / num_active
        return mean_loss, batch_loss, num_active


    def train(self, train_dataloader_dict, eval_dataloader_dict, test_dataloader_dict=None):
        """
        use data to train meta model and target model

        Args:
            train_dataloader_dict : dict of Dataloader
            eval_dataloader_dict : dict of Dataloader
        """
        self._logger.info('Start training ...')
        wait = 0
        best_epoch = -1
        best_meta_epoch = -1
        train_time = []
        eval_time = []
        train_loss_list = []
        train_acc_list = []
        eval_loss_list = []
        eval_acc_list = []
        lr_list = []

        train_loss_cla_list = []
        train_acc_cla_list = []
        eval_loss_cla_list = []
        eval_acc_cla_list = []

        train_loss_reg_list = []
        train_acc_reg_list = []
        eval_loss_reg_list = []
        eval_acc_reg_list = []

        best_train_loss = float('inf')
        best_train_loss_cla = float('inf')
        best_train_loss_reg = float('inf')
        best_train_acc = 0
        best_val_loss = float('inf')
        best_val_loss_cla = float('inf')
        best_val_loss_reg = float('inf')
        best_eval_acc = 0
        # num_batches = len(train_dataloader)
        # self._logger.info("Num_batches: train={}, eval={}".format(num_batches, len(eval_dataloader)))

        for epoch_idx in range(self.meta_train_epoch):
            start_time = time.time()
            train_loss_city_dict = self._train_epoch(train_dataloader_dict, test_dataloader_dict, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)

            self._logger.info("train source and meta complete!")

            # 在目标城市上训练及测试
            target_city = self.target_city
            # train_loss, train_acc, eval_loss, eval_acc, lr_list, best_epoch, cur_min_val_loss = self._target_ft(
            #     train_dataloader_dict[target_city], eval_dataloader_dict[target_city],
            #     test_dataloader_dict[target_city], epoch_idx)
            best_epoch_train_loss_list, best_epoch_train_loss_cla_list, best_epoch_train_loss_reg_list, best_epoch_train_acc_list, \
                best_epoch_eval_loss_list, best_epoch_eval_loss_cla_list, best_epoch_eval_loss_reg_list, best_epoch_eval_acc_list, \
                best_epoch_lr_list, cur_best_epoch, cur_min_val_loss, cur_min_val_loss_cla, cur_min_val_loss_reg = self._target_ft(
                train_dataloader_dict[target_city], eval_dataloader_dict[target_city],
                test_dataloader_dict[target_city], epoch_idx)
            if cur_min_val_loss < best_val_loss:
                #  更新最好 meta epoch 下 best epoch 状态
                best_val_loss = min(best_val_loss, cur_min_val_loss)
                best_val_loss_cla = min(best_val_loss_cla, cur_min_val_loss)
                best_val_loss_reg = min(best_val_loss_reg, cur_min_val_loss)
                best_epoch = cur_best_epoch
                best_meta_epoch = epoch_idx
                best_train_loss = min(min(best_epoch_train_loss_list), best_train_loss)
                best_train_loss_cla = min(min(best_epoch_train_loss_list), best_train_loss_cla)
                best_train_loss_reg = min(min(best_epoch_train_loss_list), best_train_loss_reg)
                best_train_acc = max(max(best_epoch_train_acc_list), best_train_acc)
                best_eval_acc = max(max(best_epoch_eval_acc_list), best_eval_acc)
                train_loss_list = best_epoch_train_loss_list
                train_loss_cla_list = best_epoch_train_loss_cla_list
                train_loss_reg_list = best_epoch_train_loss_reg_list
                train_acc_list = best_epoch_train_acc_list
                eval_loss_list = best_epoch_eval_loss_list
                eval_loss_cla_list = best_epoch_eval_loss_cla_list
                eval_loss_reg_list = best_epoch_eval_loss_reg_list
                eval_acc_list = best_epoch_eval_acc_list
                lr_list = best_epoch_lr_list

        self._logger.info(
            "Meta train finish\n best train loss:{:.3f} min val loss:{:.3f}best train acc{:.3f} best eval acc" \
                .format(best_train_loss, best_val_loss, best_train_acc, best_eval_acc))
        self.save_meta_model(self.cache_dir + '/' + self.config['model'] + '_' + "meta")  # 保存最后的状态
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch, best_meta_epoch)
        # draw best target val epoch
        self._draw_png(
            [(train_loss_list, eval_loss_list, 'loss'), (train_loss_cla_list, eval_loss_cla_list, 'loss_cla'),
             (train_loss_reg_list, eval_loss_reg_list, 'loss_reg'), (train_acc_list, eval_acc_list, 'acc'),
             (lr_list, 'lr')])
        return best_val_loss

    def _train_epoch(self, train_dataloader_dict, test_dataloader_dict, meta_epoch_idx):
        meta_model = self.model_dict["meta"]
        epoch_loss_dict = dict.fromkeys(self.train_cities, [])
        epoch_loss_cla_dict = dict.fromkeys(self.train_cities, [])
        epoch_loss_reg_dict = dict.fromkeys(self.train_cities, [])
        # epoch_loss_dict[self.target_city] = 0.0
        for c in self.train_cities:
            c_train_dataloader = train_dataloader_dict[c]
            meta_model.model.copy_invariant_params(self.model_dict[c].model)
            current_model = self.model_dict[c]
            current_model.train()
            self._logger.info("Train model on city {}".format(c))
            total_correct_cla = 0
            total_active_elements_cla = 0
            # 在每个source city上训练一个epoch
            for i, batch in tqdm(enumerate(c_train_dataloader), desc="Meta Train epoch={}".format(
                    meta_epoch_idx), total=len(c_train_dataloader)):
                X, cla_targets, cla_target_masks, reg_targets, reg_target_masks, padding_masks, batch_temporal_mat = batch
                # X: (batch_size, padded_length, feat_dim)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X = X.to(self.device)
                cla_targets = cla_targets.to(self.device)
                cla_target_masks = cla_target_masks.to(self.device)
                reg_targets = reg_targets.to(self.device)
                reg_target_masks = reg_target_masks.to(self.device)
                # padding_masks: (batch_size, padded_length, feat_dim)
                padding_masks = padding_masks.to(self.device)  # 0s: masked
                batch_temporal_mat = batch_temporal_mat.to(self.device)
                batch_interval = cul_batch_time_interval(batch_temporal_mat, padding_masks)

                graph_dict = self.multi_graph_dict[c]
                x_input = X  # 现在只输入 轨迹id序列
                # predictions = current_model(x=X)
                self.optimizer_dict[c].zero_grad()
                cla_predictions, reg_predictions = current_model(x_input, padding_masks, batch_interval, graph_dict)
                cla_targets = cla_targets[..., 0]  # (batch_size, padded_length)
                cla_target_masks = cla_target_masks[..., 0]  # (batch_size, padded_length)
                # reg_targets = reg_targets[..., 1]  # (batch_size, padded_length)
                reg_target_masks = reg_target_masks[..., 0]  # (batch_size, padded_length)
                # 计算损失
                mean_loss_cla, batch_loss_cla, num_active_cla = self._cal_cla_loss(cla_predictions, cla_targets,
                                                                                   cla_target_masks)
                mean_loss_reg, batch_loss_reg, num_active_reg = self._cal_reg_loss(reg_predictions, reg_targets,
                                                                                   reg_target_masks)
                mean_loss = self.cla_loss_weight * mean_loss_cla + self.reg_loss_weight * mean_loss_reg
                # with torch.autograd.detect_anomaly():
                mean_loss.backward()
                self.optimizer_dict[c].step()
                self.optimizer_dict[c].zero_grad()
                with torch.no_grad():
                    total_correct_cla += self._cal_cla_acc(cla_predictions, cla_targets, cla_target_masks)
                    total_active_elements_cla += num_active_cla.item()
                    epoch_loss_dict[c].append(mean_loss.item())
                    epoch_loss_cla_dict[c].append(mean_loss_cla.item())
                    epoch_loss_reg_dict[c].append(mean_loss_reg.item())
                post_fix = {
                    "mode": "Train",
                    "meta_epoch": meta_epoch_idx,
                    "current city": c,
                    "iter": i,
                    "lr": self.optimizer_dict[c].param_groups[0]['lr'],
                    "mean loss": mean_loss.item(),
                    "MlM cla loss": mean_loss_cla.item(),
                    "Loc acc(%)": total_correct_cla / total_active_elements_cla * 100,
                    "MlM reg loss": mean_loss_reg.item(),
                }
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))
            # epoch_loss_dict[c] = epoch_loss_dict[c] / total_active_elements_cla
            avg_epoch_loss = np.mean(epoch_loss_dict[c])
            avg_epoch_loss_cla = np.mean(epoch_loss_cla_dict[c])
            avg_epoch_loss_reg = np.mean(epoch_loss_reg_dict[c])
            total_correct_cla = total_correct_cla / total_active_elements_cla * 100.0
            self._logger.info(
                "Train: Meta Epoch = {}, Current city:{},avg_loss = {}, MlM cla loss = {}, MLM reg loss = {} total_acc = {}%.".format(
                    meta_epoch_idx, c, avg_epoch_loss, avg_epoch_loss_cla, avg_epoch_loss_reg, total_correct_cla))

            # 在该source city的全部测试集上算loss
            test_loss = self._valid_city_epoch_withgrad(test_dataloader_dict[c], c, meta_epoch_idx)
            self._logger.info("Meta epoch:{}, current city:{},Test loss:{:.3f}".format(meta_epoch_idx, c, test_loss))
            # 根据当前梯度 更新 metamodel （因为有些参数是共享的）
            meta_model.eval()
            for name, param in current_model.named_parameters():
                contains_specific = any(sub_str in name for sub_str in meta_model.model.domain_specific_params)
                if contains_specific:
                    continue
                assert param.grad is not None, "{},param grad is None!".format(name)
                # if param.grad is None:
                #     print(name)
                # else:
                param.data -= self.meta_lr * param.grad
        return epoch_loss_dict, epoch_loss_cla_dict, epoch_loss_reg_dict

    def _valid_city_epoch_withgrad(self, eval_dataloader, city, meta_epoch_idx, mode='Eval'):
        current_model = self.model_dict[city]
        # current_model = current_model.eval()
        # self.model = self.model.eval()
        if mode == 'Test':
            self.evaluator.clear()

        total_correct_cla = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_cla = 0  # total masked elements in epoch
        total_active_elements_reg = 0

        total_loss_cla = 0.0
        total_loss_reg = 0.0
        for i, batch in tqdm(enumerate(eval_dataloader), desc="Meta epoch :{}, {} on source city-{}".format(
                meta_epoch_idx, mode, city), total=len(eval_dataloader)):
            X, cla_targets, cla_target_masks, reg_targets, reg_target_masks, padding_masks, batch_temporal_mat = batch
            # X: (batch_size, padded_length, feat_dim)
            # batch_temporal_mat: (batch_size, padded_length, padded_length)
            X = X.to(self.device)
            cla_targets = cla_targets.to(self.device)
            cla_target_masks = cla_target_masks.to(self.device)
            reg_targets = reg_targets.to(self.device)
            reg_target_masks = reg_target_masks.to(self.device)
            # padding_masks: (batch_size, padded_length, feat_dim)
            padding_masks = padding_masks.to(self.device)  # 0s: masked
            batch_temporal_mat = batch_temporal_mat.to(self.device)
            batch_interval = cul_batch_time_interval(batch_temporal_mat, padding_masks)

            graph_dict = self.multi_graph_dict[city]
            x_input = X

            cla_predictions, reg_predictions = current_model(x_input, padding_masks, batch_interval, graph_dict)
            cla_targets = cla_targets[..., 0]  # (batch_size, padded_length)
            cla_target_masks = cla_target_masks[..., 0]  # (batch_size, padded_length)
            # reg_targets = reg_targets[..., 1]  # (batch_size, padded_length)
            reg_target_masks = reg_target_masks[..., 0]  # (batch_size, padded_length)
            # 计算损失
            mean_loss_cla, batch_loss_cla, num_active_cla = self._cal_cla_loss(cla_predictions, cla_targets,
                                                                               cla_target_masks)
            mean_loss_reg, batch_loss_reg, num_active_reg = self._cal_reg_loss(reg_predictions, reg_targets,
                                                                               reg_target_masks)
            mean_loss = self.cla_loss_weight * mean_loss_cla + self.reg_loss_weight * mean_loss_reg
            mean_loss.backward()
            with torch.no_grad():
                total_correct_cla += self._cal_cla_acc(cla_predictions, cla_targets, cla_target_masks)
                total_active_elements_cla += num_active_cla.item()
                total_active_elements_reg += num_active_reg.item()
                total_loss_cla += batch_loss_cla.item()  # add total loss of batch
                total_loss_reg += batch_loss_reg.item()  # add total loss of batch

            post_fix = {
                "mode": mode,
                "meta epoch": meta_epoch_idx,
                "current city": city,
                "iter": i,
                # "lr": self.optimizer.param_groups[0]['lr'],
                "cla loss": mean_loss_cla.item(),
                "reg loss": mean_loss_reg.item(),
                "loss": mean_loss.item(),
                "total_correct": total_correct_cla,
                "Loc acc(%)": total_correct_cla / total_active_elements_cla * 100,
            }
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))
        mean_total_loss_cla = total_loss_cla / total_active_elements_cla
        mean_total_loss_reg = total_loss_reg / total_active_elements_reg
        mean_loss_value = self.cla_loss_weight * mean_total_loss_cla + self.reg_loss_weight * mean_total_loss_reg
        total_correct = total_correct_cla / total_active_elements_cla * 100.0
        self._logger.info(
            "Train: Meta Epoch = {}, Current city:{},avg_loss = {}, cla_avg_loss={}, reg_avg_loss={} , total_acc = {}%.".format(
                meta_epoch_idx, city, mean_loss_value, mean_total_loss_cla, mean_total_loss_reg, total_correct))
        return mean_loss_value

    def _target_ft(self, train_dataloader, eval_dataloader, test_dataloader, meta_epoch_idx):
        self._logger.info("Training target model")
        min_val_loss = float('inf')
        min_val_loss_cla = float('inf')
        min_val_loss_reg = float('inf')
        wait = 0
        best_epoch = -1
        # 存储最优epoch信息
        best_epoch_train_loss_list = []
        best_epoch_train_loss_cla_list = []
        best_epoch_train_loss_reg_list = []
        best_epoch_train_acc_list = []
        best_epoch_eval_loss_list = []
        best_epoch_eval_loss_cla_list = []
        best_epoch_eval_loss_reg_list = []
        best_epoch_eval_acc_list = []
        best_epoch_lr_list = []
        # 存储当前epoch信息
        train_time = []
        eval_time = []
        train_loss = []
        train_loss_cla = []
        train_loss_reg = []
        train_acc = []
        eval_loss = []
        eval_loss_cla = []
        eval_loss_reg = []
        eval_acc = []
        lr_list = []
        meta_model = self.model_dict["meta"]
        meta_model.model.copy_invariant_params(self.model_dict[self.target_city].model)
        epochs = self.target_train_epoch
        num_batches = len(train_dataloader)
        for epoch_idx in range(epochs):
            start_time = time.time()
            train_avg_loss, train_avg_loss_cla, train_avg_loss_reg, train_avg_acc = self._tar_train_epoch(
                train_dataloader, epoch_idx, meta_epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss.append(train_avg_loss)
            train_loss_cla.append(train_avg_loss_cla)
            train_loss_reg.append(train_avg_loss_reg)
            train_acc.append(train_avg_acc)

            self._logger.info("target train epoch complete!")
            self._logger.info("evaluating now")
            # 验证
            t2 = time.time()
            eval_avg_loss, eval_avg_loss_cla, eval_avg_loss_reg, eval_avg_acc = self._valid_city_epoch(eval_dataloader,
                                                                                                       self.target_city,
                                                                                                       epoch_idx)
            end_time = time.time()
            eval_time.append(end_time - t2)
            eval_loss.append(eval_avg_loss)
            eval_loss_cla.append(eval_avg_loss_cla)
            eval_loss_reg.append(eval_avg_loss_reg)
            eval_acc.append(eval_avg_acc)
            # todo 学习率调度？
            log_lr = self.optimizer_dict[self.target_city].param_groups[0]['lr']
            lr_list.append(log_lr)
            if (epoch_idx % self.log_every) == 0:
                message = 'Epoch [{}/{}] ({})  train_loss: {:.4f}, train_loss_cla: {:.4f}, train_loss_reg: {:.4f}, \
                val_loss: {:.4f}, val_loss_cla: {:.4f}, val_loss_reg: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.target_train_epoch, (epoch_idx + 1) * num_batches, train_avg_loss, train_avg_loss_cla,
                           train_avg_loss_reg, eval_avg_loss, eval_avg_loss_cla, eval_avg_loss_reg, log_lr,
                           (end_time - start_time))
                self._logger.info(message)

            if eval_avg_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx, meta_epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving target model to {}'.format(min_val_loss, eval_avg_loss, model_file_name))
                min_val_loss = eval_avg_loss
                min_val_loss_cla = eval_avg_loss_cla
                min_val_loss_reg = eval_avg_loss_reg
                best_epoch = epoch_idx
                best_epoch_train_loss_list = train_loss
                best_epoch_train_loss_cla_list = train_loss_cla
                best_epoch_train_loss_reg_list = train_loss_reg
                best_epoch_train_acc_list = train_acc
                best_epoch_eval_loss_list = eval_loss
                best_epoch_eval_loss_cla_list = eval_loss_cla
                best_epoch_eval_loss_reg_list = eval_loss_reg
                best_epoch_eval_acc_list = eval_acc
                best_epoch_lr_list = lr_list
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

            if (epoch_idx + 1) % self.test_every == 0:
                self.evaluate(test_dataloader)
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
            # if self.load_best_epoch:
            #     self.load_model_with_epoch(best_epoch)
        min_train_loss = min(best_epoch_train_loss_list)
        min_train_loss_cla = min(best_epoch_train_loss_cla_list)
        min_train_loss_reg = min(best_epoch_train_loss_reg_list)
        best_train_acc = max(best_epoch_train_acc_list)
        best_eval_acc = max(best_epoch_eval_acc_list)
        self._logger.info("Meta epoch {} finetune complete!".format(meta_epoch_idx))
        self._logger.info(
            "Min train loss:{:.3f}, Min train cla loss:{:.3f}, Min train reg loss:{:.3f}, "
            "Min val loss{:.3f}, Min val loss_cla:{:.3f}, Min val loss_reg:{:.3f}"
            " Best train acc:{:.3f},Best eval acc:{:.3f}".format(
                min_train_loss, min_train_loss_cla, min_train_loss_reg,
                min_val_loss, min_val_loss_cla, min_val_loss_reg,
                best_train_acc,
                best_eval_acc))

        return best_epoch_train_loss_list, best_epoch_train_loss_cla_list, best_epoch_train_loss_reg_list, \
            best_epoch_train_acc_list, best_epoch_eval_loss_list, best_epoch_eval_loss_cla_list, best_epoch_eval_loss_reg_list, \
            best_epoch_eval_acc_list, best_epoch_lr_list, best_epoch, min_val_loss, min_val_loss_cla, min_val_loss_reg

    def _tar_train_epoch(self, train_dataloader, epoch_idx, meta_epoch_idx):
        c = self.target_city
        model = self.model_dict[c]
        model = model.train()

        total_correct_cla = 0
        total_active_elements_cla = 0
        total_active_elements_reg = 0
        # epoch_loss_value = 0.0
        total_loss_cla = 0.0
        total_loss_reg = 0.0
        for i, batch in tqdm(enumerate(train_dataloader), desc="Target Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            X, cla_targets, cla_target_masks, reg_targets, reg_target_masks, padding_masks, batch_temporal_mat = batch
            # X: (batch_size, padded_length, feat_dim)
            # batch_temporal_mat: (batch_size, padded_length, padded_length)
            X = X.to(self.device)
            cla_targets = cla_targets.to(self.device)
            cla_target_masks = cla_target_masks.to(self.device)
            reg_targets = reg_targets.to(self.device)
            reg_target_masks = reg_target_masks.to(self.device)
            # padding_masks: (batch_size, padded_length, feat_dim)
            padding_masks = padding_masks.to(self.device)  # 0s: masked
            batch_temporal_mat = batch_temporal_mat.to(self.device)
            batch_interval = cul_batch_time_interval(batch_temporal_mat, padding_masks)

            graph_dict = self.multi_graph_dict[c]
            x_input = X
            cla_predictions, reg_predictions = model(x_input, padding_masks, batch_interval, graph_dict)
            cla_targets = cla_targets[..., 0]  # (batch_size, padded_length)
            cla_target_masks = cla_target_masks[..., 0]  # (batch_size, padded_length)
            # reg_targets = reg_targets[..., 1]  # (batch_size, padded_length)
            reg_target_masks = reg_target_masks[..., 0]  # (batch_size, padded_length)
            # 计算损失
            mean_loss_cla, batch_loss_cla, num_active_cla = self._cal_cla_loss(cla_predictions, cla_targets,
                                                                               cla_target_masks)
            mean_loss_reg, batch_loss_reg, num_active_reg = self._cal_reg_loss(reg_predictions, reg_targets,
                                                                               reg_target_masks)
            mean_loss = self.cla_loss_weight * mean_loss_cla + self.reg_loss_weight * mean_loss_reg

            mean_loss.backward()
            self.optimizer_dict[c].step()  # todo 设置学习率调度器，暂时不用
            self.optimizer_dict[c].zero_grad()
            with torch.no_grad():
                total_correct_cla += self._cal_cla_acc(cla_predictions, cla_targets, cla_target_masks)
                total_active_elements_cla += num_active_cla.item()
                total_active_elements_reg += num_active_reg.item()
                total_loss_cla += batch_loss_cla.item()  # add total loss of batch
                total_loss_reg += batch_loss_reg.item()  # add total loss of batch
            post_fix = {
                "mode": "Train",
                "meta_epoch": meta_epoch_idx,
                "current city": c,
                "iter": i,
                "lr": self.optimizer_dict[c].param_groups[0]['lr'],
                "mlm cla loss": mean_loss_cla.item(),
                "mlm reg loss": mean_loss_reg.item(),
                "loss": mean_loss.item(),
                "acc(%)": total_correct_cla / total_active_elements_cla * 100,
            }
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))
        mean_total_loss_cla = total_loss_cla / total_active_elements_cla
        mean_total_loss_reg = total_loss_reg / total_active_elements_reg
        epoch_loss_value = self.cla_loss_weight * mean_total_loss_cla + self.reg_loss_weight * mean_total_loss_reg
        total_correct = total_correct_cla / total_active_elements_cla * 100.0
        self._logger.info(
            "Train: Meta Epoch = {}, Current city:{},avg_loss = {}, cla_avg_loss={}, reg_avg_loss={} , total_acc = {}%.".format(
                meta_epoch_idx, c, epoch_loss_value, mean_total_loss_cla, mean_total_loss_reg, total_correct))
        return epoch_loss_value, mean_total_loss_cla, mean_total_loss_reg, total_correct

    def _valid_city_epoch(self, eval_dataloader, city, epoch_idx, mode='Eval'):
        model = self.model_dict[city]
        if mode == 'Test':
            self.evaluator.clear()
        epoch_loss_cla = 0.0  # total loss of epoch
        epoch_loss_reg = 0.0 # total loss of epoch
        total_correct_cla = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_cla = 0  # total masked elements in epoch
        total_active_elements_reg = 0  # total masked elements in epoch

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc="{} on {} epoch={}".format(
                    mode, city, epoch_idx), total=len(eval_dataloader)):
                X, cla_targets, cla_target_masks, reg_targets, reg_target_masks, padding_masks, batch_temporal_mat = batch
                # X: (batch_size, padded_length, feat_dim)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X = X.to(self.device)
                cla_targets = cla_targets.to(self.device)
                cla_target_masks = cla_target_masks.to(self.device)
                reg_targets = reg_targets.to(self.device)
                reg_target_masks = reg_target_masks.to(self.device)
                # padding_masks: (batch_size, padded_length, feat_dim)
                padding_masks = padding_masks.to(self.device)  # 0s: masked
                batch_temporal_mat = batch_temporal_mat.to(self.device)
                batch_interval = cul_batch_time_interval(batch_temporal_mat, padding_masks)
                graph_dict = self.multi_graph_dict[city]
                x_input = X
                cla_predictions, reg_predictions = model(x_input, padding_masks, batch_interval, graph_dict)
                cla_targets = cla_targets[..., 0]  # (batch_size, padded_length)
                cla_target_masks = cla_target_masks[..., 0]  # (batch_size, padded_length)
                # reg_targets = reg_targets[..., 1]  # (batch_size, padded_length)
                reg_target_masks = reg_target_masks[..., 0]  # (batch_size, padded_length)
                # 计算损失
                mean_loss_cla, batch_loss_cla, num_active_cla = self._cal_cla_loss(cla_predictions, cla_targets,
                                                                                   cla_target_masks)
                mean_loss_reg, batch_loss_reg, num_active_reg = self._cal_reg_loss(reg_predictions, reg_targets,
                                                                                   reg_target_masks)
                mean_loss = self.cla_loss_weight * mean_loss_cla + self.reg_loss_weight * mean_loss_reg
                # with torch.autograd.detect_anomaly():
                if mode == 'Test':
                    evaluate_input = {
                        'loc_true': cla_targets[cla_target_masks].reshape(-1, 1).squeeze(-1).cpu().numpy(),
                        # (num_active, )
                        'loc_pred': cla_predictions[cla_target_masks].reshape(-1,
                                                                              cla_predictions.shape[-1]).cpu().numpy()
                        # (num_active, n_class)
                    }
                    self.evaluator.collect(evaluate_input)

                epoch_loss_cla += batch_loss_cla.item()  # add total loss of batch
                epoch_loss_reg += batch_loss_reg.item()  # add total loss of batch
                total_active_elements_cla += num_active_cla.item()
                total_active_elements_reg += num_active_reg.item()
                total_correct_cla += self._cal_cla_acc(cla_predictions, cla_targets, cla_target_masks)
                post_fix = {
                    "mode": mode,
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer_dict[city].param_groups[0]['lr'],
                    "loss": mean_loss.item(),
                    "cla loss": mean_loss_cla.item(),
                    "reg loss": mean_loss_reg.item(),
                    "acc(%)": total_correct_cla / total_active_elements_cla * 100,
                }
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))
            mean_epoch_loss_cla = epoch_loss_cla / total_active_elements_cla
            mean_epoch_loss_reg = epoch_loss_reg / total_active_elements_reg
            mean_epoch_loss_value = self.cla_loss_weight * mean_epoch_loss_cla + self.reg_loss_weight * mean_epoch_loss_reg
            total_correct = total_correct_cla / total_active_elements_cla * 100.0
            self._logger.info(
                "{} on target: expid = {}, Epoch = {}, avg_loss = {},cla_avg_loss={},reg_avg_loss={}, total_acc = {}%.".format(
                    mode, self.exp_id, epoch_idx, mean_epoch_loss_value, mean_epoch_loss_cla, mean_epoch_loss_reg,
                    total_correct))
            self._writer.add_scalar('{} loss'.format(mode), mean_epoch_loss_value, epoch_idx)
            self._writer.add_scalar('{} acc'.format(mode), total_correct, epoch_idx)

            if mode == 'Test':
                self.evaluator.save_result(self.evaluate_res_dir)
        return mean_epoch_loss_value, mean_epoch_loss_cla, mean_epoch_loss_reg, total_correct
