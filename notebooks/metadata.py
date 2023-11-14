SUPER_PATH = "/home/mehrshad_mirmohammadi/Project/SelfSupForRanSmooth/work_dirs/"
SUPER_PATH_SUPCON = (
    "/home/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/"
)


MODELS = {
    "SimCLR_ResNet50_CIFAR10_WithNoise_ParallelAugmentation_sigma_0.5": {
        "BASE_PATH": f"{SUPER_PATH}simclr_resnet50_cifar10_WithNoise_ParallelAugmentation_sigma_0.5/",
        "CONFIG_FILE": "{}simclr_resnet50_cifar10.py",
        "WEIGHTS": "{}epoch_600.pth",
        "PREDICTOR": "{}two_layer_512.pth",
        "CERT_PATH": "{}RS/two_layer_512_sigma_testsigma_0.5_time_2023-07-26_15:09:52.txt",
        "Extra Info": "Probability of adding noise vs original augmentation is 0.5",
        "test acc": 79.23,
        "noisy test acc": 69.27,
    },
    "SimCLR_ResNet50_CIFAR10_WithNoise_ParallelAugmentation_sigma_0.12": {
        "BASE_PATH": f"{SUPER_PATH}simclr_resnet50_cifar10_WithNoise_ParallelAugmentation_sigma_0.12/",
        "CONFIG_FILE": "{}simclr_resnet50_cifar10.py",
        "WEIGHTS": "{}epoch_600.pth",
        "PREDICTOR": "{}two_layer_512.pth",
        "CERT_PATH": "{}RS/SimCLR_with_two_layer_512_trainSigma0.12_testsigma_0.12_time_2023-07-26_09:11:31.txt",
        "Extra Info": "Probability of adding noise vs original augmentation is 0.5",
        "test acc": 89.93,
        "noisy test acc": 88.2,
    },
    "SimCLR_ResNet50_CIFAR10_WithNoise_ParallelAugmentation_sigma_0.25": {
        "BASE_PATH": f"{SUPER_PATH}simclr_resnet50_cifar10_WithNoise_ParallelAugmentation_sigma_0.25/",
        "CONFIG_FILE": "{}simclr_resnet50_cifar10.py",
        "WEIGHTS": "{}epoch_600.pth",
        "PREDICTOR": "{}two_layer_512.pth",
        "CERT_PATH": "{}RS/SimCLR_with_two_layer_512_trainSigma0.25_testsigma_0.25_time_2023-07-25_12:58:48.txt",
        "Extra Info": "Probability of adding noise vs original augmentation is 0.5",
        "test acc": 84,
        "noisy test acc": 77,
    },
    "Original_SimCLR_ResNet50_CIFAR10_sigma0.25": {
        "BASE_PATH": f"{SUPER_PATH}simclr_resnet50_cifar10/",
        "CONFIG_FILE": "{}simclr_resnet50_cifar10.py",
        "WEIGHTS": "{}epoch_1000.pth",
        "PREDICTOR": "{}sigma0.25_two_layer_512.pth",
        "CERT_PATH": "{}RS/SimCLR_with_two_layer_512_trainSigma0.25_testsigma_0.25_time_2023-07-27_11:33:20.txt",
        "Extra Info": "Note that the encoder is not trained using noise, but only the decoder head is trained using noise",
        "test acc": 88.7,
        "noisy test acc": 80.3,
    },
    "Original_SimCLR_ResNet50_CIFAR10_sigma0.5": {
        "BASE_PATH": f"{SUPER_PATH}simclr_resnet50_cifar10/",
        "CONFIG_FILE": "{}simclr_resnet50_cifar10.py",
        "WEIGHTS": "{}epoch_1000.pth",
        "PREDICTOR": "{}sigma0.5_two_layer_512.pth",
        "CERT_PATH": "{}RS/SimCLR_with_two_layer_512_trainSigma0.5_testsigma_0.5_time_2023-07-27_13:20:06.txt",
        "Extra Info": "Note that the encoder is not trained using noise, but only the decoder head is trained using noise",
        "test acc": 81,
        "noisy test acc": 62,
    },
    "SimCLR_ResNet50_CIFAR10_WithNoise_SequentialAugmentation_sigma_0.25": {
        "BASE_PATH": f"{SUPER_PATH}simclr_resnet50_cifar10_WithNoise_SequentialAugmentation_sigma_0.25/",
        "CONFIG_FILE": "{}simclr_resnet50_cifar10.py",
        "WEIGHTS": "{}epoch_600.pth",
        "PREDICTOR": "{}two_layer_512.pth",
        "CERT_PATH": "{}RS/SimCLR_with_two_layer_512_trainSigma0.25_testsigma_0.25_time_2023-07-27_11:20:43.txt",
        "Extra Info": "",
        "test acc": 75.3,
        "noisy test acc": 75.9,
    },
    "SupCon_ResNet50_CIFAR10_WithNoise_ParallelAugmentation_sigma_0.25": {
        "BASE_PATH": f"{SUPER_PATH_SUPCON}SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm_sigma_0.25/",
        "CONFIG_FILE": "{}???",
        "WEIGHTS": "{}ckpt_epoch_1000.pth",
        # "PREDICTOR": "{}two_layer_512.pth",
        # "CERT_PATH": "{}RS/testsigma_0.25_time_2023-08-01_08:39:31.txt",
        # "Extra Info": "",
        # "test acc": 93.0,  # 93.7
        # "noisy test acc": 82.0,  # 86.7
        "PREDICTOR": "{}one_layer.pth",
        "CERT_PATH": "{}RS/testsigma_0.25_time_2023-08-01_08:39:31.txt",
        "Extra Info": "",
        "test acc": 93.6,
        "noisy test acc": 85.3,
    },
    "SupCon_ResNet110_CIFAR10_WithNoise_ParallelAugmentation_sigma_0.25": {
        "BASE_PATH": f"{SUPER_PATH_SUPCON}SupCon_cifar10_resnet110_lr_0.5_decay_0.0001_bsz_4096_temp_0.1_trial_0_cosine_warm_sigma_0.25/",
        "CONFIG_FILE": "{}???",
        "WEIGHTS": "{}ckpt_epoch_1000.pth",
        # "PREDICTOR": "{}two_layer_64.pth",
        # "CERT_PATH": "{}RS/???",
        # "Extra Info": "",
        # "test acc": 90.8,
        # "noisy test acc": 82.5,
        "PREDICTOR": "{}one_layer.pth",
        "CERT_PATH": "{}RS/???",
        "Extra Info": "",
        "test acc": 90.7,
        "noisy test acc": 82.0,
    },
}

for model in MODELS.keys():
    MODELS[model]["CONFIG_FILE"] = MODELS[model]["CONFIG_FILE"].format(
        MODELS[model]["BASE_PATH"]
    )
    MODELS[model]["WEIGHTS"] = MODELS[model]["WEIGHTS"].format(
        MODELS[model]["BASE_PATH"]
    )
    MODELS[model]["PREDICTOR"] = MODELS[model]["PREDICTOR"].format(
        MODELS[model]["BASE_PATH"]
    )
    MODELS[model]["CERT_PATH"] = MODELS[model]["CERT_PATH"].format(
        MODELS[model]["BASE_PATH"]
    )
