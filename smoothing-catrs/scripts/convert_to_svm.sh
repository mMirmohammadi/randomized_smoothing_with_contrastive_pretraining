
CUDA_VISIBLE_DEVICES=0 python code/convert_to_svm.py --model_path logs/cifar10/supcon/model_2l.pth
CUDA_VISIBLE_DEVICES=0 python code/convert_to_svm.py --model_path logs/cifar10/supcon/model_3l.pth

CUDA_VISIBLE_DEVICES=2 python code/convert_to_svm.py --sigma 0.5 --model_path "logs/cifar10/supcon/noise_0.5/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth"
CUDA_VISIBLE_DEVICES=3 python code/convert_to_svm.py --sigma 1.0 --model_path "logs/cifar10/supcon/noise_1.0/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth"
