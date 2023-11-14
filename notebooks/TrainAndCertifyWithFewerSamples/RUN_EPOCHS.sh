#!/bin/bash
#SBATCH -n 2
#SBATCH -G a100-40g:2
#SBATCH --cpus-per-task 6
#SBATCH --mem-per-cpu 6G

# print the local id and name of the GPU node on which this job is running
srun -n 2 -G a100-40g:2 --cpus-per-task 6 --mem-per-cpu 6G --gpu-bind per_task:1 -- python batched_run_consistency_finetune.py --batch_size 5000 --model_path /scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth \
  --freeze_epochs 8 \
  --freeze_lr 0.01 \
  --unfreeze_epochs 8 \
  --unfreeze_lr 0.001