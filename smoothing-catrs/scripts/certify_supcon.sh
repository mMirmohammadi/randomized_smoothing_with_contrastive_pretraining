
CUDA_VISIBLE_DEVICES=1 python code/certify.py cifar10 logs/cifar10/supcon/model_2l_svm3.pth 0.25 \
test/certify/cifar10/supcon/2l_2/noise_0.25.tsv --N 100000 --skip 1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.25/1e5.pt

CUDA_VISIBLE_DEVICES=1 python code/certify.py cifar10 logs/cifar10/supcon/model_3l_svm.pth 0.25 \
test/certify/cifar10/supcon/3l/noise_0.25.tsv --N 100000 --skip 1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.25/1e5.pt


CUDA_VISIBLE_DEVICES=2 python code/certify.py cifar10 "logs/cifar10/supcon/noise_0.5/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last_svm_n.pth" 0.5 \
test/certify/cifar10/supcon/2l/noise_0.5.tsv --N 100000 --skip 1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/0.5/1e5.pt

CUDA_VISIBLE_DEVICES=3 python code/certify.py cifar10 "logs/cifar10/supcon/noise_1.0/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last_svm_n.pth" 1.0 \
test/certify/cifar10/supcon/2l/noise_1.0.tsv --N 100000 --skip 1 \
--save-predictions --load-noise /mnt/disks/rs-cert-data/noise/cifar10/gaussian/1.0/1e5.pt
