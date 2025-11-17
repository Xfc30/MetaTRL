import os
import time
import numpy as np
import torch

from libcity.executor import MetaTRLExecutor
import matplotlib as mpl

from libcity.utils.utils import cul_batch_time_interval

mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


class MetaTRLLMExecutor(MetaTRLExecutor):
    # model:dict {name,model}
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.criterion = torch.nn.NLLLoss(ignore_index=0, reduction='none')
        self.initial_ckpt = self.config.get("initial_ckpt", None)
        self.unload_param = self.config.get("unload_param", [])

        self.time_interval_scale = self.config.get("time_interval_scale", 300)
        self.max_interval = self.config.get("max_time_scale_s", 5000)
        self.time_interval_scale_list = self.config.get("time_interval_scales", [])

        self.target_city = self.config.get("dataset", None)
        # 也许需要使用来减小显存
        # ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}["float16"]
        # self.ctx = torch.autocast(device_type='cuda', dtype=ptdtype)

        if self.initial_ckpt:
            self.load_model_with_initial_ckpt(self.initial_ckpt)


    def _train_epoch(self, train_dataloader_dict, test_dataloader_dict, meta_epoch_idx):
        meta_model = self.model_dict["meta"]
        epoch_loss_dict = dict.fromkeys(self.train_cities, 0.0)
        # epoch_loss_dict[self.target_city] = 0.0
        for opt in self.optimizer_dict.values():
            opt.zero_grad()
        for c in self.train_cities:
            c_train_dataloader = train_dataloader_dict[c]
            meta_model.model.copy_invariant_params(self.model_dict[c].model)
            current_model = self.model_dict[c]
            current_model.train()
            self._logger.info("Train model on city {}".format(c))
            total_correct = 0
            total_active_elements = 0
            # 在每个source city上训练一个epoch
            for i, batch in tqdm(enumerate(c_train_dataloader), desc="Meta Train epoch={}".format(
                    meta_epoch_idx), total=len(c_train_dataloader)):
                X, targets, target_masks, padding_masks, batch_temporal_mat = batch
                # X: (batch_size, padded_length, feat_dim)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X = X.to(self.device)
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)
                # padding_masks: (batch_size, padded_length, feat_dim)
                padding_masks = padding_masks.to(self.device)  # 0s: masked
                batch_temporal_mat = batch_temporal_mat.to(self.device)
                batch_interval = cul_batch_time_interval(batch_temporal_mat,padding_masks)

                graph_dict = self.multi_graph_dict[c]
                x_input = X  # 现在只输入 轨迹id序列
                # self.optimizer_dict[c].zero_grad()
                predictions = current_model(x_input,padding_masks,batch_interval,graph_dict)  # todo frequency参数优化
                targets = targets[..., 0]  # (batch_size, padded_length)
                target_masks = target_masks[..., 0]  # (batch_size, padded_length)
                # 计算损失
                batch_loss_list = self.criterion(predictions.transpose(1,2), targets)
                batch_loss = torch.sum(batch_loss_list)
                num_active = target_masks.sum()  # 有效预测位置的数量
                mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization

                # with torch.autograd.detect_anomaly():
                mean_loss.backward()
                self.optimizer_dict[c].step()  # todo 设置学习率调度器，暂时不用
                self.optimizer_dict[c].zero_grad()
                with torch.no_grad():
                    predictions_label = targets[target_masks]
                    predictions_output = predictions[target_masks].argmax(dim=-1)
                    assert predictions_label.shape == predictions_output.shape
                    correct = predictions_label.eq(predictions_output).sum().item()
                    total_correct += correct
                    total_active_elements += num_active.item()
                    epoch_loss_dict[c] += batch_loss.item()
                assert total_active_elements !=0, "total_active_elements={}, num_active={}".format(total_active_elements,num_active.item())
                post_fix = {
                    "mode": "Train",
                    "meta_epoch": meta_epoch_idx,
                    "current city": c,
                    "iter": i,
                    "lr": self.optimizer_dict[c].param_groups[0]['lr'],
                    "loss": mean_loss.item(),
                    "acc(%)": total_correct / total_active_elements * 100,
                }
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))
            epoch_loss_dict[c] = epoch_loss_dict[c] / total_active_elements
            total_correct = total_correct / total_active_elements * 100.0
            self._logger.info("Train: Meta Epoch = {}, Current city:{},avg_loss = {}, total_acc = {}%.".format(
                meta_epoch_idx, c, epoch_loss_dict[c], total_correct))

            # 在该source city的全部测试集上算loss
            test_loss = self._valid_city_epoch_withgrad(test_dataloader_dict[c], c, meta_epoch_idx)
            self._logger.info("Meta epoch:{}, current city:{},Test loss:{:.3f}".format(meta_epoch_idx, c, test_loss))
            # 根据当前梯度 更新 metamodel （因为有些参数是共享的）
            meta_model.eval()
            # todo 是否需要在_valid_city_epoch_withgrad调用 backward，以使用测试集上的梯度更新meta model
            # todo  现在的方案 保持跟cola代码一致
            # todo  cola 论文跟代码这里不一致
            for name, param in current_model.named_parameters():
                contains_specific = any(sub_str in name for sub_str in meta_model.model.domain_specific_params)
                if contains_specific:
                    continue
                assert param.grad is not None,"{},param grad is None!".format(name)
                # if param.grad is None:
                #     print(name)
                # else:
                param.data -= self.meta_lr * param.grad
        return epoch_loss_dict

    def _valid_city_epoch_withgrad(self, eval_dataloader, city, meta_epoch_idx, mode='Eval'):
        current_model = self.model_dict[city]
        # current_model = current_model.eval()
        # self.model = self.model.eval()
        if mode == 'Test':
            self.evaluator.clear()

        total_correct = 0  # total top@1 acc for masked elements in epoch
        total_active_elements = 0  # total masked elements in epoch

        total_loss = 0.0
        for i, batch in tqdm(enumerate(eval_dataloader), desc="Meta epoch :{}, {} on source city-{}".format(
                meta_epoch_idx, mode, city), total=len(eval_dataloader)):
            X, targets, target_masks, padding_masks, batch_temporal_mat = batch
            # X: (batch_size, padded_length, feat_dim)
            # batch_temporal_mat: (batch_size, padded_length, padded_length)
            X = X.to(self.device)

            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)
            # padding_masks: (batch_size, padded_length, feat_dim)
            padding_masks = padding_masks.to(self.device)  # 0s: masked
            batch_temporal_mat = batch_temporal_mat.to(self.device)
            batch_interval = cul_batch_time_interval(batch_temporal_mat, padding_masks)

            graph_dict = self.multi_graph_dict[city]
            x_input = X
            # predictions = current_model(x=X)
            # predictions, batch_loss_list = current_model(x_input, targets)  # todo frequency参数优化
            predictions = current_model(x_input,padding_masks,batch_interval,graph_dict)  # todo frequency参数优化
            # 计算损失
            targets = targets[..., 0]  # (batch_size, padded_length)
            target_masks = target_masks[..., 0]  # (batch_size, padded_length)
            batch_loss_list = self.criterion(predictions.transpose(1, 2), targets)
            batch_loss = torch.sum(batch_loss_list)
            num_active = target_masks.sum()
            mean_loss = batch_loss / num_active
            mean_loss.backward()

            with torch.no_grad():
                correct = predictions.argmax(dim=-1).eq(targets).sum().item()
                total_correct += correct
                total_active_elements += num_active.item()
                total_loss += batch_loss.item()  # add total loss of batch

        mean_loss_value = total_loss / total_active_elements
        post_fix = {
            "mode": mode,
            "meta epoch": meta_epoch_idx,
            "current city": city,
            "iter": i,
            # "lr": self.optimizer.param_groups[0]['lr'],
            "loss": mean_loss_value,
            "total_correct": total_correct,
            "acc(%)": total_correct / total_active_elements * 100,
        }
        if i % self.log_batch == 0:
            self._logger.info(str(post_fix))
        return mean_loss_value

    def _target_ft(self, train_dataloader, eval_dataloader, test_dataloader, meta_epoch_idx):
        self._logger.info("Training target model")
        min_val_loss = float('inf')
        wait = 0
        best_epoch = -1
        # 存储最优epoch信息
        best_epoch_train_loss_list = []
        best_epoch_train_acc_list = []
        best_epoch_eval_loss_list = []
        best_epoch_eval_acc_list = []
        best_epoch_lr_list = []
        # 存储当前epoch信息
        train_time = []
        eval_time = []
        train_loss = []
        train_acc = []
        eval_loss = []
        eval_acc = []
        lr_list = []
        meta_model = self.model_dict["meta"]
        meta_model.model.copy_invariant_params(self.model_dict[self.target_city].model)
        epochs = self.target_train_epoch
        num_batches = len(train_dataloader)
        for epoch_idx in range(epochs):
            start_time = time.time()
            train_avg_loss, train_avg_acc = self._tar_train_epoch(train_dataloader, epoch_idx, meta_epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss.append(train_avg_loss)
            train_acc.append(train_avg_acc)
            self._logger.info("target train epoch complete!")
            self._logger.info("evaluating now")
            # 验证
            t2 = time.time()
            eval_avg_loss, eval_avg_acc = self._valid_city_epoch(eval_dataloader, self.target_city, epoch_idx)
            end_time = time.time()
            eval_time.append(end_time - t2)
            eval_loss.append(eval_avg_loss)
            eval_acc.append(eval_avg_acc)
            # todo 学习率调度？
            log_lr = self.optimizer_dict[self.target_city].param_groups[0]['lr']
            lr_list.append(log_lr)
            if (epoch_idx % self.log_every) == 0:
                message = 'Epoch [{}/{}] ({})  train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, (epoch_idx + 1) * num_batches, train_avg_loss,
                           eval_avg_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if eval_avg_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx, meta_epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving target model to {}'.format(min_val_loss, eval_avg_loss, model_file_name))
                min_val_loss = eval_avg_loss
                best_epoch = epoch_idx
                best_epoch_train_loss_list = train_loss
                best_epoch_train_acc_list = train_acc
                best_epoch_eval_loss_list = eval_loss
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
        best_train_acc = max(best_epoch_train_acc_list)
        best_eval_acc = max(best_epoch_eval_acc_list)
        self._logger.info("Meta epoch {} finetune complete!".format(meta_epoch_idx))
        self._logger.info(
            "Min train loss:{:.3f},Min val loss{:.3f}\nBest train acc:{:.3f},Best eval acc:{:.3f}".format(min_train_loss,
                                                                                                         min_val_loss,
                                                                                                         best_train_acc,
                                                                                                         best_eval_acc))

        return best_epoch_train_loss_list, best_epoch_train_acc_list, best_epoch_eval_loss_list, best_epoch_eval_acc_list, best_epoch_lr_list, best_epoch, min_val_loss

    def _tar_train_epoch(self, train_dataloader, epoch_idx, meta_epoch_idx):
        c = self.target_city
        model = self.model_dict[c]
        model = model.train()

        total_correct = 0
        total_active_elements = 0
        epoch_loss_value = 0.0
        for i, batch in tqdm(enumerate(train_dataloader), desc="Target Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            X, targets, target_masks, padding_masks, batch_temporal_mat = batch
            # X: (batch_size, padded_length, feat_dim)
            # batch_temporal_mat: (batch_size, padded_length, padded_length)
            X = X.to(self.device)
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)
            # padding_masks: (batch_size, padded_length, feat_dim)
            padding_masks = padding_masks.to(self.device)  # 0s: masked
            batch_temporal_mat = batch_temporal_mat.to(self.device)
            batch_interval = cul_batch_time_interval(batch_temporal_mat, padding_masks)

            graph_dict = self.multi_graph_dict[c]
            x_input = X
            predictions = model(x_input,padding_masks,batch_interval,graph_dict)  # todo frequency参数优化
            # 计算损失
            targets = targets[..., 0]  # (batch_size, padded_length)
            target_masks = target_masks[..., 0]  # (batch_size, padded_length)
            batch_loss_list = self.criterion(predictions.transpose(1, 2), targets)

            batch_loss = torch.sum(batch_loss_list)
            num_active = target_masks.sum()  # 有效预测位置的数量
            mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization

            # with torch.autograd.detect_anomaly():
            mean_loss.backward()
            self.optimizer_dict[c].step()  # todo 设置学习率调度器，暂时不用
            self.optimizer_dict[c].zero_grad()
            with torch.no_grad():
                correct = predictions.argmax(dim=-1).eq(targets).sum().item()
                total_correct += correct
                total_active_elements += num_active.item()
                epoch_loss_value += batch_loss.item()
            post_fix = {
                "mode": "Train",
                "meta_epoch": meta_epoch_idx,
                "current city": c,
                "iter": i,
                "lr": self.optimizer_dict[c].param_groups[0]['lr'],
                "loss": mean_loss.item(),
                "acc(%)": total_correct / total_active_elements * 100,
            }
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))
        epoch_loss_value = epoch_loss_value / total_active_elements
        total_correct = total_correct / total_active_elements * 100.0
        self._logger.info("Train: Meta Epoch = {}, Current city:{},avg_loss = {}, total_acc = {}%.".format(
            meta_epoch_idx, c, epoch_loss_value, total_correct))
        return epoch_loss_value, total_correct

    def _valid_city_epoch(self, eval_dataloader, city, epoch_idx, mode='Eval'):
        model = self.model_dict[city]
        if mode == 'Test':
            self.evaluator.clear()
        epoch_loss = 0  # total loss of epoch
        total_correct = 0  # total top@1 acc for masked elements in epoch
        total_active_elements = 0  # total masked elements in epoch

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc="{} on {} epoch={}".format(
                    mode, city,epoch_idx), total=len(eval_dataloader)):
                X, targets, target_masks, padding_masks, batch_temporal_mat = batch
                # X: (batch_size, padded_length, feat_dim)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X = X.to(self.device)
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)
                # padding_masks: (batch_size, padded_length, feat_dim)
                padding_masks = padding_masks.to(self.device)  # 0s: masked
                batch_temporal_mat = batch_temporal_mat.to(self.device)
                batch_interval = cul_batch_time_interval(batch_temporal_mat,padding_masks)
                graph_dict = self.multi_graph_dict[city]
                x_input = X
                predictions = model(x_input,padding_masks,batch_interval,graph_dict)  # todo frequency参数优化
                targets = targets[..., 0]  # (batch_size, padded_length)
                target_masks = target_masks[..., 0]  # (batch_size, padded_length)
                # 计算损失
                batch_loss_list = self.criterion(predictions.transpose(1, 2), targets)
                if mode == 'Test':
                    evaluate_input = {
                        'loc_true': targets[target_masks].reshape(-1, 1).squeeze(-1).cpu().numpy(),  # (num_active, )
                        'loc_pred': predictions[target_masks].reshape(-1, predictions.shape[-1]).cpu().numpy()
                        # (num_active, n_class)
                    }
                    self.evaluator.collect(evaluate_input)
                batch_loss = torch.sum(batch_loss_list)
                num_active = target_masks.sum()  # 有效预测位置的数量
                mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization

                mask_label = targets[target_masks]  # (num_active, )
                lm_output = predictions[target_masks].argmax(dim=-1)  # (num_active, )
                assert mask_label.shape == lm_output.shape
                correct = mask_label.eq(lm_output).sum().item()
                total_correct += correct
                total_active_elements += num_active.item()
                epoch_loss += batch_loss.item()  # add total loss of batch

                post_fix = {
                    "mode": mode,
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer_dict[city].param_groups[0]['lr'],
                    "loss": mean_loss.item(),
                    "acc(%)": total_correct / total_active_elements * 100,
                }
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))

            epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
            total_correct = total_correct / total_active_elements * 100.0
            self._logger.info("{} on target: expid = {}, Epoch = {}, avg_loss = {}, total_acc = {}%.".format(
                mode, self.exp_id, epoch_idx, epoch_loss, total_correct))
            self._writer.add_scalar('{} loss'.format(mode), epoch_loss, epoch_idx)
            self._writer.add_scalar('{} acc'.format(mode), total_correct, epoch_idx)

            if mode == 'Test':
                self.evaluator.save_result(self.evaluate_res_dir)
        return epoch_loss, total_correct

