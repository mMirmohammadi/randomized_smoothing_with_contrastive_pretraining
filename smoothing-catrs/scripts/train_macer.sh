dev=2

# it seems the default parameters used are bs=64 lr=0.01 and 440 epochs??

CUDA_VISIBLE_DEVICES=3 python code/train_macer.py cifar10 cifar_resnet110 --lr 0.01 --batch 64 --lr_step_size 200 --epochs 440  --noise 0.25 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 12.0 --id 2
CUDA_VISIBLE_DEVICES=4 python code/train_macer.py cifar10 cifar_resnet110 --lr 0.01 --batch 64 --lr_step_size 200 --epochs 440  --noise 0.5 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 4.0 --id 2
CUDA_VISIBLE_DEVICES=5 python code/train_macer.py cifar10 cifar_resnet110 --lr 0.01 --batch 64 --lr_step_size 200 --epochs 440  --noise 1.0 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 12.0 --deferred --id 2