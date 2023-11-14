dev=3

# # CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/64/0/checkpoint.pth.tar 0.25 \
# # test/certify/cifar10/macer/num_16/lbd_16.0/gamma_8.0/beta_16.0/256/0/noise_0.25.tsv --N=100000 --skip=1 \
# # --save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.25/1e5.pt

# CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/macer/num_16/lbd_6.0/gamma_8.0/beta_16.0/noise_0.5/cifar_resnet110/64/0/checkpoint.pth.tar 0.5 \
# test/certify/cifar10/macer/num_16/lbd_6.0/gamma_8.0/beta_16.0/256/0/noise_0.5.tsv --N=100000 --skip=1 \
# --save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.5/1e5.pt

# CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/macer_deferred50/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_1.0/cifar_resnet110/64/0/checkpoint.pth.tar 1.0 \
# test/certify/cifar10/macer_deferred50/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_1.0.tsv --N=100000 --skip=1 \
# --save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/1.0/1e5.pt

CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/64/1/checkpoint.pth.tar 0.25 \
test/certify/cifar10/macer/num_16/lbd_16.0/gamma_8.0/beta_16.0/64/1/noise_0.25.tsv --N=100000 --skip=1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.25/1e5.pt

CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/macer/num_16/lbd_6.0/gamma_8.0/beta_16.0/noise_0.5/cifar_resnet110/64/1/checkpoint.pth.tar 0.5 \
test/certify/cifar10/macer/num_16/lbd_6.0/gamma_8.0/beta_16.0/64/1/noise_0.5.tsv --N=100000 --skip=1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.5/1e5.pt

# CUDA_VISIBLE_DEVICES=$dev python code/certify.py cifar10 logs/cifar10/macer_deferred50/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_1.0/cifar_resnet110/64/1/checkpoint.pth.tar 1.0 \
# test/certify/cifar10/macer_deferred50/num_16/lbd_12.0/gamma_8.0/beta_16.0/64/1/noise_1.0.tsv --N=100000 --skip=1 \
# --save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/1.0/1e5.pt

CUDA_VISIBLE_DEVICES=0 python code/certify.py cifar10 logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/64/2/checkpoint.pth.tar 0.25 \
test/certify/cifar10/macer/num_16/lbd_16.0/gamma_8.0/beta_16.0/64/2/noise_0.25.tsv --N=100000 --skip=1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.25/1e5.pt

CUDA_VISIBLE_DEVICES=1 python code/certify.py cifar10 logs/cifar10/macer/num_16/lbd_4.0/gamma_8.0/beta_16.0/noise_0.5/cifar_resnet110/64/2/checkpoint.pth.tar 0.5 \
test/certify/cifar10/macer/num_16/lbd_4.0/gamma_8.0/beta_16.0/64/2/noise_0.5.tsv --N=100000 --skip=1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.5/1e5.pt

CUDA_VISIBLE_DEVICES=2 python code/certify.py cifar10 logs/cifar10/macer_deferred200/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_1.0/cifar_resnet110/64/2/checkpoint.pth.tar 1.0 \
test/certify/cifar10/macer_deferred50/num_16/lbd_12.0/gamma_8.0/beta_16.0/64/2/noise_1.0.tsv --N=100000 --skip=1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/1.0/1e5.pt
