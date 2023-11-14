dev=1

# it seems the authors used batch=400
net=cifar_resnet20

CUDA_VISIBLE_DEVICES=$dev python code/train_cohen.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0

CUDA_VISIBLE_DEVICES=$dev python code/train_cohen.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.5 --id 0

CUDA_VISIBLE_DEVICES=$dev python code/train_cohen.py cifar10 $net --lr 0.1 --lr_step_size 50 --epochs 150 --noise 1.0 --id 0

