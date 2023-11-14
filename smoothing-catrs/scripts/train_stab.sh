dev=5

# No idea what batch size to use. It seems the paper repo only contains testing code...

CUDA_VISIBLE_DEVICES=$dev python code/train_stab.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 \
--lbd 2.0 --id 0
CUDA_VISIBLE_DEVICES=$dev python code/train_stab.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.5 \
--lbd 2.0 --id 0
CUDA_VISIBLE_DEVICES=$dev python code/train_stab.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 1.0 \
--lbd 1.0 --id 0