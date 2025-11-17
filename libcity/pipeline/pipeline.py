import copy
import random
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.model.trajectory_embedding import MetaTRL
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name, config_file, saved_model, train, other_args)  # config_file?

    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 1000000)
        config['exp_id'] = exp_id
    logger = get_logger(config)
    logger.info('Begin pretrain-pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    seed = config.get('seed', 0)
    set_random_seed(seed)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()  # dataloader对象
    data_feature = dataset.get_data_feature()
    model_cache_file = config.get('model_cache_file', './libcity/cache/{}/{}/model_cache/{}_{}_{}.pt'.format(model_name,
                                                                                                             exp_id,
                                                                                                             exp_id,
                                                                                                             model_name,
                                                                                                             dataset_name))
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)

    initial_ckpt = config.get("initial_ckpt", None)
    pretrain_path = config.get("pretrain_path", None)
    if train:
        executor.train(train_data, valid_data, test_data)
        # 在最相似轨迹搜索任务中，train->base_dataset valid->detour_dataset test->query_dataset
        if saved_model:
            executor.save_model(model_cache_file)
        executor.evaluate(test_data)  # if patch pretraining, no evaluate
    else:
        # assert os.path.exists(model_cache_file) or initial_ckpt is not None or pretrain_path is not None
        if initial_ckpt is None and pretrain_path is None:
            executor.load_model_state(model_cache_file)
        executor.evaluate(test_data)


def run_model_cross(task=None, model_name=None, target_dataset_name=None, config_file=None,
                    saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        target_dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, target_dataset_name, config_file, saved_model, train,
                          other_args)  # config_file?

    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 1000000)
        config['exp_id'] = exp_id
    logger = get_logger(config)
    logger.info('Begin pretrain-pipeline, task={}, model_name={}, target_dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(target_dataset_name), str(exp_id)))
    logger.info(config.config)
    seed = config.get('seed', 0)
    set_random_seed(seed)
    train_cities = config.get('train_cities', None)
    assert train_cities is not None, "train_cities is not defined"
    data_feature_dict = {}
    train_data_loader_dict = {}  # 三元组词典
    valid_data_loader_dict = {}  # 三元组词典
    test_data_loader_dict = {}  # 三元组词典
    meta_vocab_size = 0
    meta_max_time_scale_s = 0
    meta_time_interval_scales = []
    # 加载所有的数据集
    # 加载target数据集
    for idx, city in enumerate(train_cities + [target_dataset_name]):
        config_city = copy.deepcopy(config)
        config_city.set("dataset", city)  # 修改跟数据集相关的配置
        #   "roadnetwork": "porto_roadmap_edge_porto_True_1_merge",
        #   "geo_file": "porto_roadmap_edge_porto_True_1_merge_withdegree",
        #   "rel_file": "porto_roadmap_edge_porto_True_1_merge_withdegree",
        config_city.set("roadnetwork", city + "_roadmap_edge_" + city + "_True_1_merge")
        config_city.set("geo_file", city + "_roadmap_edge_" + city + "_True_1_merge_withdegree")
        config_city.set("rel_file", city + "_roadmap_edge_" + city + "_True_1_merge_withdegree")
        config_city.set("struct_features_file", city + "_roadmap_edge_" + city + "_True_1_merge_structure_features.geo")
        if idx< len(train_cities) +1 -1 :  # 目标城市不需要处理
            config_city.set("max_train_size", None)  # 源城市不限制训练集大小
            config_city.set("max_time_scale_s", config.get("train_cities_max_time_scale_s", [3000])[idx])
            config_city.set("time_interval_scales", config.get("train_cities_time_interval_list", [[15, 60, 150]])[idx])
        logger.info("max train size of {} : {}".format(city,config_city.get("max_train_size")))
        dataset = get_dataset(config_city)
        train_data, valid_data, test_data = dataset.get_data()  # dataloader对象
        train_data_loader_dict[city] = train_data
        valid_data_loader_dict[city] = valid_data
        test_data_loader_dict[city] = test_data
        data_feature = dataset.get_data_feature()
        data_feature_dict[city] = data_feature
        meta_vocab_size = max(meta_vocab_size, data_feature.get("vocab_size"))
        c_max_time_scale_s=config_city.get("max_time_scale_s", 0)
        if config_city.get("max_time_scale_s",0)>meta_max_time_scale_s:
            meta_max_time_scale_s = c_max_time_scale_s
            meta_time_interval_scales = config_city.get("train_cities_time_interval_list",[])



    meta_data_feature = {'vocab_size': meta_vocab_size,'node_fea_dim':45} # 43
    config_meta = copy.deepcopy(config)
    config_meta.set("max_time_scale_s",meta_max_time_scale_s)
    config_meta.set("time_interval_scales",meta_time_interval_scales)
    data_feature_dict["meta"] = meta_data_feature

    model_cache_file = config.get('model_cache_file', './libcity/cache/{}/{}/model_cache/{}_{}_{}.pt'.format(model_name,
                                                                                                             exp_id,
                                                                                                             exp_id,
                                                                                                             model_name,
                                                                                                             target_dataset_name))
    model_dict = {}
    # meta model
    model_dict["meta"] = get_model(config_meta, meta_data_feature)
    for i in range(len(train_cities) + 1):
        if i != len(train_cities):
            data_name = train_cities[i]
        else:
            data_name = target_dataset_name
        # set COLAConfig
        # model_args = dict(seed=args.seed, data=data, datapath=args.datapath, domain_specific_params=args.domain_specific_params, n_linear=args.n_linear, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len, use_start_letter=args.use_start_letter, start_letter=args.start_letter, device=device, n_layer=args.n_layer_t, n_head=args.n_head_t, n_embd=args.n_embd, block_size=args.block_size, bias=args.bias, vocab_size=data_neural[data]['num_locs'], token_size=data_neural[data]['num_locs']+1, dropout=args.dropout, meta_lr=args.meta_lr, update_lr=args.update_lr, meta_epochs=args.meta_epochs, city_epochs=args.city_epochs, test_epochs=args.test_epochs)
        model = get_model(config, data_feature_dict[data_name])
        # model = COLA(COLAConfig(**model_args)).to(device)
        model_dict[data_name] = model
        # optim_dict[data_name] = optim.Adam(model.parameters(), lr=model.config.update_lr)

    # model = get_model(config, data_feature)
    executor = get_executor(config, model_dict, data_feature_dict)

    initial_ckpt = config.get("initial_ckpt", None)
    pretrain_path = config.get("pretrain_path", None)
    if train:
        executor.train(train_data_loader_dict, valid_data_loader_dict, test_data_loader_dict)
        # 在最相似轨迹搜索任务中，train->base_dataset valid->detour_dataset test->query_dataset
        if saved_model:  # 保存target model
            executor.save_model(model_cache_file)
        executor.evaluate(test_data_loader_dict[target_dataset_name])  # if patch pretraining, no evaluate
    else:
        # assert os.path.exists(model_cache_file) or initial_ckpt is not None or pretrain_path is not None
        if initial_ckpt is None and pretrain_path is None:
            executor.load_model_state(model_cache_file)
        executor.evaluate(test_data_loader_dict)
