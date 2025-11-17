# MetaTRL


## Model
The model's code is in libcity/model/trajectory_embedding/MetaTRL. All json files are Configuration Files.

The files in libcity/data, libcity/executor, libcity/evaluator are used for data loading, training and evaluation respectively.
## Requirements
Our code is based on python3.9 and pytorch 2.6.

## Data

Due to file size limitations, we are unable to upload the datasets. You can obtain them from the paper  **START**.

## Run
Please run the file run_model_cross.py with arguments for pre-training meta learning. 

For example, pretrain MetaTRL:
```shell
python run_model_cross.py --model MetaMulLMLearning --dataset porto --train_cities bj --config porto_cross_metafeats --gpu_id 0 --distribution geometric --avg_mask_len 2 --time_masking_ratio 0.2

```



After gain the pre-training model, you can fine-tune to downstream task, such as for TTE:

```shell
python run_model.py --model  MetaTRLLinearETA --dataset porto --config porto_cola --gpu_id 0 --pretrain_path ./libcity/cache/COLAMetaLearning/${exp_id}/model_cache/${exp_id}_MetaMulLMLearning_porto.pt
```

