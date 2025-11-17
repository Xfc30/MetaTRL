import os

import numpy as np
import torch
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from libcity.utils import ensure_dir
from libcity.executor.scheduler import CosineLRScheduler
from libcity.utils import get_evaluator


class AbstractMultiSourceExecutor(object):

    def __init__(self, config, model_dict, data_feature_dict):
        self.config = config
        # self.data_feature = data_feature
        self.data_feature_dict = data_feature_dict
        self.model_dict = model_dict
        self._logger = getLogger()

        self.train_cities = self.config.get("train_cities", None)
        self.target_city = self.config.get("dataset", None)
        # self.vocab_size = self.data_feature.get('vocab_size')
        # self.usr_num = self.data_feature.get('usr_num')
        self.meta_train_epoch = self.config.get("meta_train_epoch", 5)
        self.target_train_epoch = self.config.get("target_train_epoch", 20)
        self.exp_id = self.config.get('exp_id', None)
        self.device = self.config.get('device', torch.device('cpu'))
        self.epochs = self.config.get('max_epoch', 100)
        self.model_name = self.config.get('model', '')
        # 优化器
        self.learner = self.config.get('learner', 'adamw')
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.grad_accmu_steps = self.config.get('grad_accmu_steps', 1)
        self.test_every = self.config.get('test_every', 5)

        self.meta_lr = self.config.get('meta_lr', 5e-4)

        self.lr_decay = self.config.get('lr_decay', True)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'cosinelr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.lr_warmup_epoch = self.config.get("lr_warmup_epoch", 5)
        self.lr_warmup_init = self.config.get("lr_warmup_init", 1e-6)
        self.t_in_epochs = self.config.get("t_in_epochs", True)

        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.log_batch = self.config.get('log_batch', 10)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.l2_reg = self.config.get('l2_reg', None)
        # self.adj_mx = self.data_feature.get('adj_mx')
        # self.node_features = self.data_feature.get('node_features')
        # self.edge_index = self.data_feature.get('edge_index')
        # self.loc_trans_prob = self.data_feature.get('loc_trans_prob')
        self.add_lap = self.config.get('add_lap', True)
        # self.hyper_graph = self.data_feature.get('hyper_graph')
        # self.graph_dict = {
        #     'node_features': self.node_features,
        #     'edge_index': self.edge_index,
        #     'loc_trans_prob': self.loc_trans_prob,
        # }
        self.multi_graph_dict = {}
        for c in self.train_cities:
            node_features = self.data_feature_dict[c].get('node_features')
            node_struct_features = self.data_feature_dict[c].get('node_struct_features')
            edge_index = self.data_feature_dict[c].get('edge_index')
            loc_trans_prob = self.data_feature_dict[c].get('loc_trans_prob')
            graph_dict = {
                'node_features': node_features,
                'node_struct_features': node_struct_features,
                'edge_index': edge_index,
                'loc_trans_prob': loc_trans_prob,
            }
            self.multi_graph_dict[c] = graph_dict
        self.multi_graph_dict[self.target_city] = {
            'node_features': self.data_feature_dict[self.target_city].get('node_features'),
            'node_struct_features': self.data_feature_dict[self.target_city].get('node_struct_features'),
            'edge_index': self.data_feature_dict[self.target_city].get('edge_index'),
            'loc_trans_prob': self.data_feature_dict[self.target_city].get('loc_trans_prob'),}
        self.cache_dir = './libcity/cache/{}/{}/model_cache'.format(self.model_name, self.exp_id)
        self.png_dir = './libcity/cache/{}/{}'.format(self.model_name, self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/{}/evaluate_cache'.format(self.model_name, self.exp_id)
        self.summary_writer_dir = './libcity/cache/{}/{}'.format(self.model_name, self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.png_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)
        self._writer = SummaryWriter(self.summary_writer_dir)

        # self.model = model.to(self.device)  # bertlm
        for model in self.model_dict.values():
            model.to(self.device)
        self._logger.info("meta model is as follows:")
        self._logger.info(model_dict["meta"])
        for name, param in self.model_dict["meta"].named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model_dict["meta"].parameters()])
        self._logger.info('Total parameter numbers of meta model: {}'.format(total_num))

        self.optimizer_dict = self._build_optimizer_dict()
        # self.lr_scheduler = self._build_lr_scheduler()  # todo 先不用
        for optimizer in self.optimizer_dict.values():
            optimizer.zero_grad()

        self.evaluator = get_evaluator(self.config, self.data_feature_dict[self.target_city])  # todo 加载评估器

    def save_model(self, cache_name):
        """
        将当前的目标城市模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        c = self.target_city
        ensure_dir(self.cache_dir)
        config = dict()
        config['model'] = self.model_dict[c].cpu()
        config['optimizer_state_dict'] = self.optimizer_dict[c].state_dict()
        torch.save(config, cache_name)
        self.model_dict[c].to(self.device)
        self._logger.info("Saved model at " + cache_name)

    def save_meta_model(self, cache_name):
        c = "meta"
        ensure_dir(self.cache_dir)
        config = dict()
        config['model'] = self.model_dict[c].cpu()
        # config['optimizer_state_dict'] = self.optimizer_dict[c].state_dict() # meta 没有优化器
        torch.save(config, cache_name)
        self.model_dict[c].to(self.device)
        self._logger.info("Saved meta model at " + cache_name)

    # todo 修正
    def load_model_state(self, cache_name):
        """
        加载对应目标城市模型的 cache （用于加载参数直接进行测试的场景）

        Args:
            cache_name(str): 保存的文件名
        """
        c = self.target_city
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model_dict[c].load_state_dict(checkpoint['model'].state_dict())
        self.optimizer_dict[c].load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at " + cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at " + cache_name)

    def save_model_with_epoch(self, epoch, meta_epoch):
        """
        保存某个epoch的目标城市模型

        Args:
            epoch(int): 轮数
        """
        c = self.target_city
        ensure_dir(self.cache_dir)
        config = dict()
        config['model'] = self.model_dict[c].cpu()
        config['optimizer_state_dict'] = self.optimizer_dict[c].state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + c + '_me{:d}_'.format(
            meta_epoch) + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self.model_dict[c].to(self.device)
        self._logger.info("Saved model at me{}_e{}".format(meta_epoch, epoch))
        return model_path

    def load_model_with_epoch(self, epoch, meta_epoch):
        """
        加载某个epoch的目标城市模型

        Args:
            epoch(int): 轮数
        """
        c = self.target_city
        # model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + c + '_me{:d}_'.format(
            meta_epoch) + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model_dict[c] = checkpoint['model'].to(self.device)
        self.optimizer_dict[c].load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at me{}_e{}".format(meta_epoch, epoch))

    def _build_optimizer_dict(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        optimizer_dict = {}
        if self.learner.lower() == 'adam':
            for city_name in self.train_cities:
                optimizer = torch.optim.Adam(self.model_dict[city_name].parameters(), lr=self.learning_rate,
                                             eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
                optimizer_dict[city_name] = optimizer
            optimizer_dict[self.target_city] = torch.optim.Adam(self.model_dict[self.target_city].parameters(),
                                                                lr=self.learning_rate,
                                                                eps=self.lr_epsilon, betas=self.lr_betas,
                                                                weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            for city_name in self.train_cities:
                optimizer = torch.optim.AdamW(self.model_dict[city_name].parameters(), lr=self.learning_rate,
                                              eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
                optimizer_dict[city_name] = optimizer
            optimizer_dict[self.target_city] = torch.optim.AdamW(self.model_dict[self.target_city].parameters(),
                                                                 lr=self.learning_rate,
                                                                 eps=self.lr_epsilon, betas=self.lr_betas,
                                                                 weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            for city_name in self.train_cities:
                optimizer = torch.optim.SGD(self.model_dict[city_name].parameters(), lr=self.learning_rate,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
                optimizer_dict[city_name] = optimizer
            optimizer_dict[self.target_city] = torch.optim.SGD(self.model_dict[self.target_city].parameters(),
                                                               lr=self.learning_rate,
                                                               momentum=self.lr_momentum,
                                                               weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            for city_name in self.train_cities:
                optimizer = torch.optim.Adagrad(self.model_dict[city_name].parameters(), lr=self.learning_rate,
                                                eps=self.lr_epsilon, weight_decay=self.weight_decay)
                optimizer_dict[city_name] = optimizer
            optimizer_dict[self.target_city] = torch.optim.Adagrad(self.model_dict[self.target_city].parameters(),
                                                                   lr=self.learning_rate,
                                                                   eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            for city_name in self.train_cities:
                optimizer = torch.optim.RMSprop(self.model_dict[city_name].parameters(), lr=self.learning_rate,
                                                alpha=self.lr_alpha, eps=self.lr_epsilon,
                                                momentum=self.lr_momentum, weight_decay=self.weight_decay)
                optimizer_dict[city_name] = optimizer
            optimizer_dict[self.target_city] = torch.optim.RMSprop(self.model_dict[self.target_city].parameters(),
                                                                   lr=self.learning_rate,
                                                                   alpha=self.lr_alpha, eps=self.lr_epsilon,
                                                                   momentum=self.lr_momentum,
                                                                   weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            for city_name in self.train_cities:
                optimizer = torch.optim.SparseAdam(self.model_dict[city_name].parameters(), lr=self.learning_rate,
                                                   eps=self.lr_epsilon, betas=self.lr_betas)
                optimizer_dict[city_name] = optimizer
            optimizer_dict[self.target_city] = torch.optim.SparseAdam(self.model_dict[self.target_city].parameters(),
                                                                      lr=self.learning_rate,
                                                                      eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            for city_name in self.train_cities:
                optimizer = torch.optim.Adam(self.model_dict[city_name].parameters(), lr=self.learning_rate,
                                             eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
                optimizer_dict[city_name] = optimizer
            optimizer_dict[self.target_city] = torch.optim.Adam(self.model_dict[self.target_city].parameters(),
                                                                lr=self.learning_rate,
                                                                eps=self.lr_epsilon, betas=self.lr_betas,
                                                                weight_decay=self.weight_decay)
        return optimizer_dict

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            elif self.lr_scheduler_type.lower() == 'cosinelr':
                lr_scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=self.epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                    warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init, t_in_epochs=self.t_in_epochs)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def _lamda_scheduler(self, lamda, niter_per_ep):
        start_warmup_value = 8 / lamda
        base_value = 1 / lamda
        epochs = self.epochs
        warmup_epochs = self.lr_warmup_epoch
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor train not implemented")

    def _train_epoch(self, train_dataloader_dict, test_dataloader_dict, meta_epoch_idx):
        raise NotImplementedError("Executor evaluate not implemented")

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        raise NotImplementedError("Executor evaluate not implemented")

    def _valid_city_epoch(self, eval_dataloader, city, epoch_idx, mode='Eval'):
        raise NotImplementedError("Executor evaluate not implemented")

    def _draw_png(self, data):
        raise NotImplementedError("Executor evaluate not implemented")

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor evaluate not implemented")
