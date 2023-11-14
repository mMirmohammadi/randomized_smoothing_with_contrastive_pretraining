dev=0

# CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/cohen/noise_0.25/cifar_resnet110/256/1/checkpoint.pth.tar 0.25 \
# test/certify/cifar10/cohen/256/0/noise_0.25.tsv --N=100000 --skip=1 \
# --save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.25/1e5.pt

CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/cohen/noise_0.5/cifar_resnet110/256/1/checkpoint.pth.tar 0.5 \
test/certify/cifar10/cohen/256/0/noise_0.5.tsv --N=100000 --skip=1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.5/1e5.pt

CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/cohen/noise_1.0/cifar_resnet110/256/1/checkpoint.pth.tar 1.0 \
test/certify/cifar10/cohen/256/0/noise_1.0.tsv --N=100000 --skip=1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/1.0/1e5.pt
