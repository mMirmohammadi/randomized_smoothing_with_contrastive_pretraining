python run_finetune.py --model_path /home/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_1_cosine_warm/ckpt_epoch_1000.pth \
  --freeze_epochs 8 \
  --freeze_lr 0.01 \
  --unfreeze_epochs 8 \
  --unfreeze_lr 0.001