dev=2

# bs=256 lr=0.1
net=cifar_resnet20

CUDA_VISIBLE_DEVICES=$dev python code/train_salman.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 \
--attack PGD --epsilon 256.0 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 0
CUDA_VISIBLE_DEVICES=$dev python code/train_salman.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.5 \
--attack PGD --epsilon 256.0 --num-steps 10 --warmup 10 --num-noise-vec 8 --id 0
CUDA_VISIBLE_DEVICES=$dev python code/train_salman.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150  --noise 1.0 \
--attack PGD --epsilon 512.0 --num-steps 10 --warmup 20 --num-noise-vec 4 --id 0


### These are not trained???
# CUDA_VISIBLE_DEVICES=$dev python code/train_salman.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 1.0 \
# --attack PGD --epsilon 512.0 --num-steps 10 --warmup 20 --num-noise-vec 4 --id 2
# CUDA_VISIBLE_DEVICES=$dev python code/train_salman.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 1.0 \
# --attack PGD --epsilon 256.0 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 3