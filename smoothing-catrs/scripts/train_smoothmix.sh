dev=4

# bs 256 lr 0.1 (lr 0.01 for mnist)
net=cifar_resnet20

CUDA_VISIBLE_DEVICES=$dev python code/train_smoothmix.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 \
--num-noise-vec 2 --num-steps 4 --alpha 0.5 --mix_step 0 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=$dev python code/train_smoothmix.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.5 \
--num-noise-vec 2 --num-steps 4 --alpha 1.0 --mix_step 1 --eta 5.0 --id 0
CUDA_VISIBLE_DEVICES=$dev python code/train_smoothmix.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 --noise 1.0 \
--num-noise-vec 2 --num-steps 4 --alpha 2.0 --mix_step 1 --eta 5.0 --id 0
