# Self-supervised Cross-city Trajectory Representation Learning based on Meta-Learning （AAAI-2026 Accepted）
The Implementation of Self-supervised Cross-city Trajectory Representation Learning based on Meta-Learning (MetaTRL)
## Implementation
The code is based on LibCity. For datail, you can learn more about in [LibCity](https://github.com/LibCity/Bigscity-LibCity).
The model's code is in libcity/model/*. All json files are Configuration Files.
## Run
Please run the file run_model_cross.py with arguments. 
```shell
python run_model_cross.py --model MetaMulLMLearning --dataset porto --config porto_cross_metafeats --gpu_id 0 --split true --masking_ratio 0.2 --distribution geometric
```
