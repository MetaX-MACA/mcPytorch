1. Create and activate conda env with python version 3.8.16

2. Install mcPytorch wheel and mcVision wheel

3. Set ImageNet2012 datasets path in the file <config_dataset_path.py>.
    There are 3 different size of datasets: normal, small and tiny.
    We need to set these 3 datasets paths separately.

4. Run resnet50 test with different config.
    Parameter:
        --batch_size (Default: 256)
        --total_epoch (Default: 20)
        --learn_rate (Default: 1e-4)
        --amp_mode (Default: False)
        --ddp_mode (Default: False)
        --world_size (Default: -1)
        --dataset_mode (Default: tiny)
    
    For example:
    * FP32_TinyDataset:
        - python resnet50_imagenet.py --batch_size 256 --total_epoch 20 --learn_rate 1e-4 --dataset_mode tiny
    * AMP_TinyDataset:
        - python resnet50_imagenet.py --batch_size 256 --total_epoch 20 --learn_rate 1e-4 --amp_mode --dataset_mode tiny
    * FP32_DDP_TinyDataset (8 cards):
            - python resnet50_imagenet.py --batch_size 256 --total_epoch 20 --learn_rate 1e-4 --ddp_mode --world_size 8 --dataset_mode tiny
    * AMP_DDP_TinyDataset (8 cards):
         - python resnet50_imagenet.py --batch_size 256 --total_epoch 20 --learn_rate 1e-4 --amp_mode --ddp_mode --world_size 8 --dataset_mode tiny
