# Implementation of ResNet for CIFAR-10 classification

## Requirements
- Python 3.6+
- PyTorch 1.6.0+

## Usage
1. Train

```
mkdir path/to/checkpoint_dir
python train.py --n 3 --checkpoint_dir path/to/checkpoint_dir
```
`n` means the network depth, you can choose from {3, 5, 7, 9}, which means ResNet-{20, 32, 44, 56}.
For other options, please refer helps: `python train.py -h`.
When you run the code for the first time, the dataset will be downloaded automatically.

2. Test

When your training is done, the model parameter file `path/to/checkpoint_dir/model_final.pth` will be generated.
```
python test.py --n 3 --params_path path/to/checkpoint_dir/model_final.pth
```

## Note
If you want to specify GPU to use, you should set environment variable `CUDA_VISIBLE_DEVICES=0`, for example.

## References
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep residual learning for image recognition," In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.

## Modify
```
conda activate your_env

python train.py --n 3 --n_iter 64000 --optimizer sgd --lr 0.01 --save_params_freq 10000 --checkpoint_dir checkpoint_sgd_0.01 | Tee-Object -FilePath logs/sgd_lr0.01.log
python train.py --n 3 --n_iter 64000 --optimizer sgd --lr 0.1 --save_params_freq 10000 --checkpoint_dir checkpoint_sgd_0.1 | Tee-Object -FilePath logs/sgd_lr0.1.log
python train.py --n 3 --n_iter 64000 --optimizer sgd --lr 0.5 --save_params_freq 10000 --checkpoint_dir checkpoint_sgd_0.5 | Tee-Object -FilePath logs/sgd_lr0.5.log

python train.py --n 3 --n_iter 64000 --optimizer momentum --lr 0.01 --save_params_freq 10000 --checkpoint_dir checkpoint_momentum_0.01 | Tee-Object -FilePath logs/momentum_lr0.01.log
python train.py --n 3 --n_iter 64000 --optimizer momentum --lr 0.1 --save_params_freq 10000 --checkpoint_dir checkpoint_momentum_0.1 | Tee-Object -FilePath logs/momentum_lr0.1.log
python train.py --n 3 --n_iter 64000 --optimizer momentum --lr 0.5 --save_params_freq 10000 --checkpoint_dir checkpoint_momentum_0.5 | Tee-Object -FilePath logs/momentum_lr0.5.log

python train.py --n 3 --n_iter 64000 --optimizer adam --lr 0.01 --save_params_freq 10000 --checkpoint_dir checkpoint_adam_0.01 | Tee-Object -FilePath logs/adam_lr0.01.log
python train.py --n 3 --n_iter 64000 --optimizer adam --lr 0.001 --save_params_freq 10000 --checkpoint_dir checkpoint_adam_0.001 | Tee-Object -FilePath logs/adam_lr0.001.log
python train.py --n 3 --n_iter 64000 --optimizer adam --lr 0.0005 --save_params_freq 10000 --checkpoint_dir checkpoint_adam_0.0005 | Tee-Object -FilePath logs/adam_lr0.0005.log
python train.py --n 3 --n_iter 64000 --optimizer adam --lr 0.0001 --save_params_freq 10000 --checkpoint_dir checkpoint_adam_0.0001 | Tee-Object -FilePath logs/adam_lr0.0001.log
```