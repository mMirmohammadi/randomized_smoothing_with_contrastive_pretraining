dev=1

# default parameters: lr=0.01, bs=256

CUDA_VISIBLE_DEVICES=$dev python code/train_consistency.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 \
    --num-noise-vec 2 --lbd 20 --id 0
CUDA_VISIBLE_DEVICES=$dev python code/train_consistency.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.5 \
    --num-noise-vec 2 --lbd 10 --id 0
CUDA_VISIBLE_DEVICES=$dev python code/train_consistency.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 1.0 \
    --num-noise-vec 2 --lbd 10 --id 0