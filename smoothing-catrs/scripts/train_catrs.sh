dev=1

# CUDA_VISIBLE_DEVICES=$dev python code/smooth_prediction.py cifar10 logs/cifar10/cohen/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar \
#     0.25 test/smooth_prediction/cifar10/cohen/0/noise_train_0.25.tsv --N=10000 --skip=1 --split=train

net=cifar_resnet20

CUDA_VISIBLE_DEVICES=$dev python code/train_catrs.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 \
--num-noise-vec 4 --noise 0.25 --id 0 --eps 256.0 --num-steps 4 --lbd 0.5
CUDA_VISIBLE_DEVICES=$dev python code/train_catrs.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 \
--num-noise-vec 4 --noise 0.5 --id 0 --eps 256.0 --num-steps 4 --lbd 1.0
CUDA_VISIBLE_DEVICES=$dev python code/train_catrs.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 \
--num-noise-vec 4 --noise 1.0 --id 0 --eps 256.0 --num-steps 4 --lbd 2.0