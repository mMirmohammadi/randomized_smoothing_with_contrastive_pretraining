ssh gcp-us-1

find . -type f -name "ckpt_epoch_*.pth" -exec rm {} \;

######### Train 2Level [1,2] with different batch sizes ########
cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/SupContrast && clear

srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy.py --batch_size 500 --n_aug_view 1 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 2

srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy.py --batch_size 550 --n_aug_view 1 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 2

srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy.py --batch_size 512 --n_aug_view 1 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 2 --trial RedoBatchSize512

srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy.py --batch_size 512 --n_aug_view 1 --num_workers 16 --learning_rate 0.5 --temp 0.1 --cosine --alpha 1 0 2 --trial RedoBatchSize512AndConsEpochs200Epochs --epochs 200

srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy.py --batch_size 512 --n_aug_view 1 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 2 --model resnet101

srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy_pullOnly.py --batch_size 512 --n_aug_view 1 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 32

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/SupContrast && clear && srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy_pullOnly.py --batch_size 512 --n_aug_view 1 \
  --num_workers 16 --learning_rate 0.5 --temp 0.1 --cosine --epochs 200 --trial Only200Epochs \
  --positive_loss log --positive_loss_temperature 0.5 --alpha 1 0 1


srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy_with_Consistency.py --batch_size 512 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 2 \
  --consistency_ratio 0.999

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/notebooks/SVM/ && clear

rsync -azPu hala:project /scratch/mehrshad_mirmohammadi/ --exclude='*wandb*' --exclude='*__pycache__*' --exclude='*.git*' --exclude='*.pth*' --exclude='*tensorboard*' -n
rsync -azPu /scratch/mehrshad_mirmohammadi/project/ hala:/home/mehrshad_mirmohammadi/project/ --exclude='*wandb*' --exclude='*__pycache__*' --exclude='*.git*' --exclude='*tensorboard*' --exclude='*ckpt_epoch*' --exclude='*dataset_cache*' -n

rsync -azPu hala:project/smoothing-catrs /scratch/mehrshad_mirmohammadi/project --exclude='*wandb*' --exclude='*__pycache__*' --exclude='*.git*' --exclude='*tensorboard*' -n

###### Train different baselines with different batch sizes

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_cohen.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0 --workers 8

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_stab.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --lbd 2.0 --id 0 --workers 8

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_salman.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 --attack PGD --epsilon 256.0 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 0 --workers 8

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_macer.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 --num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 12.0 --id 0 --workers 8

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_consistency.py cifar10 cifar_resnet110 --lr 0.05 --lr_step_size 50 --epochs 150  --noise 0.25 --num-noise-vec 2 --lbd 20 --id 0 --workers 8

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_smoothmix.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --num-noise-vec 2 --num-steps 4 --alpha 0.5 --mix_step 0 --eta 5.0 --id 0 --workers 8

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_catrs.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --num-noise-vec 4 --noise 0.25 --id 0 --eps 256.0 --num-steps 4 --lbd 0.5 --workers 8

## batch half

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_cohen.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0 --workers 8 --batch 128

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_stab.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --lbd 2.0 --id 0 --workers 8 --batch 128

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_salman.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 --attack PGD --epsilon 256.0 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 0 --workers 8 --batch 128

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_macer.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 --num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 12.0 --id 0 --workers 8 --batch 32

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_consistency.py cifar10 cifar_resnet110 --lr 0.05 --lr_step_size 50 --epochs 150  --noise 0.25 --num-noise-vec 2 --lbd 20 --id 0 --workers 8 --batch 128

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_smoothmix.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --num-noise-vec 2 --num-steps 4 --alpha 0.5 --mix_step 0 --eta 5.0 --id 0 --workers 8 --batch 128

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_catrs.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --num-noise-vec 4 --noise 0.25 --id 0 --eps 256.0 --num-steps 4 --lbd 0.5 --workers 8 --batch 128

## batch twice

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_cohen.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --id 0 --workers 8 --batch 512

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_stab.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --lbd 2.0 --id 0 --workers 8 --batch 512

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_salman.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 --attack PGD --epsilon 256.0 --num-steps 10 --warmup 10 --num-noise-vec 4 --id 0 --workers 8 --batch 512

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_macer.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150  --noise 0.25 --num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 12.0 --id 0 --workers 8 --batch 128

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_consistency.py cifar10 cifar_resnet110 --lr 0.025 --lr_step_size 50 --epochs 150  --noise 0.25 --num-noise-vec 2 --lbd 20 --id 0 --workers 8 --batch 512

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_smoothmix.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --noise 0.25 --num-noise-vec 2 --num-steps 4 --alpha 0.5 --mix_step 0 --eta 5.0 --id 0 --workers 8 --batch 512

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/train_catrs.py cifar10 cifar_resnet110 --lr 0.1 --lr_step_size 50 --epochs 150 --num-noise-vec 4 --noise 0.25 --id 0 --eps 256.0 --num-steps 4 --lbd 0.5 --workers 8 --batch 512

## certification

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/256/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/512/0/checkpoint.pth.tar 0.25 ./logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/512/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/128/0/checkpoint.pth.tar 0.25 ./logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/128/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/cohen/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/cohen/noise_0.25/cifar_resnet110/256/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/cohen/noise_0.25/cifar_resnet110/512/0/checkpoint.pth.tar 0.25 ./logs/cifar10/cohen/noise_0.25/cifar_resnet110/512/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/cohen/noise_0.25/cifar_resnet110/128/0/checkpoint.pth.tar 0.25 ./logs/cifar10/cohen/noise_0.25/cifar_resnet110/128/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/128/0/checkpoint.pth.tar 0.25 ./logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/128/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/64/0/checkpoint.pth.tar 0.25 ./logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/64/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/32/0/checkpoint.pth.tar 0.25 ./logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/32/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar_resnet110/256/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar_resnet110/512/0/checkpoint.pth.tar 0.25 ./logs/cifar10/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar_resnet110/512/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar_resnet110/128/0/checkpoint.pth.tar 0.25 ./logs/cifar10/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar_resnet110/128/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/stab/lbd_2.0/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/stab/lbd_2.0/noise_0.25/cifar_resnet110/256/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/stab/lbd_2.0/noise_0.25/cifar_resnet110/512/0/checkpoint.pth.tar 0.25 ./logs/cifar10/stab/lbd_2.0/noise_0.25/cifar_resnet110/512/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/stab/lbd_2.0/noise_0.25/cifar_resnet110/128/0/checkpoint.pth.tar 0.25 ./logs/cifar10/stab/lbd_2.0/noise_0.25/cifar_resnet110/128/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/256/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/512/0/checkpoint.pth.tar 0.25 ./logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/512/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/128/0/checkpoint.pth.tar 0.25 ./logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/128/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/256/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/512/0/checkpoint.pth.tar 0.25 ./logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/512/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/128/0/checkpoint.pth.tar 0.25 ./logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/128/0/noise_0.25.tsv --N=10000 --skip=10 --batch=10000 --alpha=0.01"



tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/256/0/noise_0.25_original.tsv --N=100000 --skip=10 --batch=10000 --alpha=0.001"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/cohen/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/cohen/noise_0.25/cifar_resnet110/256/0/noise_0.25_original.tsv --N=100000 --skip=10 --batch=10000 --alpha=0.001"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/64/0/checkpoint.pth.tar 0.25 ./logs/cifar10/macer/num_16/lbd_12.0/gamma_8.0/beta_16.0/noise_0.25/cifar_resnet110/64/0/noise_0.25_original.tsv --N=100000 --skip=10 --batch=10000 --alpha=0.001"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/salman/pgd_256.0_10_10/num_4/noise_0.25/cifar_resnet110/256/0/noise_0.25_original.tsv --N=100000 --skip=10 --batch=10000 --alpha=0.001"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/stab/lbd_2.0/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/stab/lbd_2.0/noise_0.25/cifar_resnet110/256/0/noise_0.25_original.tsv --N=100000 --skip=10 --batch=10000 --alpha=0.001"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/256/0/noise_0.25_original.tsv --N=100000 --skip=10 --batch=10000 --alpha=0.001"

tmux new-session -d "cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-catrs && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 8 --mem 25g -- python code/certify.py cifar10 ./logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar 0.25 ./logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/256/0/noise_0.25_original.tsv --N=100000 --skip=10 --batch=10000 --alpha=0.001"


## Ensemble

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/notebooks/Ensemble && clear && srun -w gcp-eu-2 -n1 -G a100-40g:2 -c 10 --mem 20g -- python run_ensemble_diffModels.py --batch_size 10000 --model_paths ../../smoothing-catrs/logs/cifar10/smix_0.5_4_m0/eta_5.0/num_2/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar ../../smoothing-catrs/logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar ../../smoothing-catrs/logs/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/noise_0.25/cifar_resnet110/256/0/checkpoint.pth.tar --tag "Consistency_SmoothMix_CatRS"

##########################################################

##
cd /scratch/mehrshad_mirmohammadi/project/notebooks/SVM/ && clear && srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_svm.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet34_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_1_nNoise_2_alpha_[1.0, 0.0, 2.0]_cosine_warm/last.pth" --kernel linear --C 0.01

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/notebooks/SVM/ && clear && srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_svm.py --batch_size 10000 --kernel linear --C 0.01 --model_path "/scratch/mehrshad_mirmohammadi/project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth"

cd /scratch/mehrshad_mirmohammadi/project/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_consistency_finetune.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet34_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_1_nNoise_2_alpha_[1.0, 0.0, 2.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001
##

cd /scratch/mehrshad_mirmohammadi/project/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_consistency_finetune.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/project/SupContrast/save/HiSupCon/cifar10_models/OnlyPull_log_0.5_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Only200Epochs_nAug_1_nNoise_2_alpha_[1.0, 0.0, 16.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/notebooks/SVM/ && clear && srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_weighted_svm.py --batch_size 10000 --kernel linear --C 0.01 --model_path "/scratch/mehrshad_mirmohammadi/project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth"

cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/smoothing-consistency/ && clear && srun -w gcp-us-1 -n1 -G a100-40g:1 -c 12 --mem 80g -- python code/certify_svm.py cifar10 logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.05/checkpoint.pth.tar 0.25 logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.05/certification_noisy_svm.txt --batch=5000 --N=10000 --skip=10 --alpha=0.01


cd /scratch/mehrshad_mirmohammadi/ && source anaconda3/bin/activate && cd project/notebooks/Ensemble && clear && srun -w gcp-us-1 -n1 -G a100-40g:2 -c 10 --mem 20g -- python run_ensemble_diffModels.py --batch_size 10000 --model_paths  --tag ""

############

rsync -azPu hala:project /scratch/mehrshad_mirmohammadi/ --exclude='*wandb*' && echo Sync Finished && echo && srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_svm.py --batch_size 10000 --model_path "../../SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_1_nNoise_2_alpha_[1.0, 0.0, 2.0]_cosine_warm" --kernel linear --C 0.01

cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -w gcp-eu-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_svm.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/OnlyPullOtherThanClass_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_2_nNoise_2_alpha_[1.0, 0.0, 2.0]_cosine_warm/last.pth" --kernel linear --C 0.01



# Train Hierarchical Contrastive Learning
srun -n1 -G a100-40g:2 -c 20 --mem 150g -- python main_supcon_hierarchy.py --batch_size 512 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --trial Sequential-BenchMarking \
  --alpha 1 1 1

# ceritfy Cohen
srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python code/certify.py cifar10 logs/cifar10/cohen/noise_0.25/cifar_resnet110/9507/checkpoint.pth.tar 0.25 logs/cifar10/cohen/noise_0.25/cifar_resnet110/9507/certification.txt --batch=5000 --N=10000 --skip=10 --alpha=0.01

# srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python code/train_consistency.py cifar10 cifar_resnet110 --workers 12 --noise_sd 0.25  --num-noise-vec 2 --lbd 20.0 -eta 0.5

# train consistency
srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python code/train_consistency.py cifar10 cifar_resnet110 --workers 12 --noise_sd 0.25  --num-noise-vec 2 --lbd 20.0 --eta 0.5 --lr 0.05

# sync files
rsync -azP hala:Project /scratch/mehrshad_mirmohammadi/
rsync -aznP /scratch/mehrshad_mirmohammadi/Project/ hala:Project/
rsync -azP /scratch/mehrshad_mirmohammadi/Project/ hala:Project/
rsync -azv hala:Project /scratch/mehrshad_mirmohammadi/ --exclude='*ckpt_epoch*' --exclude='*.png' --exclude='*.zip' --exclude='*smoothing-ensemble-models*' --exclude='*wandb*'

rsync -aznP hala:project /scratch/mehrshad_mirmohammadi/ --exclude='*wandb*'

rsync -aznP /scratch/mehrshad_mirmohammadi/project hala:project --exclude='*wandb*'

# Interactive Bash
srun -n1 -G a100-40g:1 -c 12 --mem 80g --pty /bin/bash

# Certify Cohen full
srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python code/certify.py cifar10 logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.01/checkpoint.pth.tar 0.25 logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.01/certification_Full.txt --batch=20000 --N=100000 --skip=10 --alpha=0.001

# Certify Consistency
srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python code/certify.py cifar10 logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.05/checkpoint.pth.tar 0.25 logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet110/lr0.05/certification.txt --batch=20000 --N=100000 --skip=10 --alpha=0.001


# Train SupCon Using Consistency
srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_consistency.py --batch_size 5000 --model_path /scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth

srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_consistency.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth"

#Train and finetune using consistency
srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_consistency_finetune.py --batch_size 5000 --model_path /scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth \
  --freeze_epochs 8 \
  --freeze_lr 0.01 \
  --unfreeze_epochs 8 \
  --unfreeze_lr 0.001

srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_2_nNoise_2_alpha_[1.0, 1.0, 1.0]_cosine_warm/last.pth" \
  --freeze_epochs 8 \
  --freeze_lr 0.01 \
  --unfreeze_epochs 8 \
  --unfreeze_lr 0.001

srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_2_nNoise_2_alpha_[3.0, 2.0, 1.0]_cosine_warm/last.pth" \
  --freeze_epochs 8 \
  --freeze_lr 0.01 \
  --unfreeze_epochs 8 \
  --unfreeze_lr 0.001

srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_2_nNoise_2_cosine_warm/last.pth" \
  --freeze_epochs 8 \
  --freeze_lr 0.01 \
  --unfreeze_epochs 8 \
  --unfreeze_lr 0.001

srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/last.pth" \
  --freeze_epochs 8 \
  --freeze_lr 0.01 \
  --unfreeze_epochs 8 \
  --unfreeze_lr 0.001

########

srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" \
  --freeze_epochs 8 \
  --freeze_lr 0.01 \
  --unfreeze_epochs 8 \
  --unfreeze_lr 0.001

cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_svm_FineTune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth"   --freeze_epochs 8   --freeze_lr 0.01   --unfreeze_epochs 8   --unfreeze_lr 0.001

##############################################################################
# SVM + SupCon
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth" --kernel linear --C 0.1

# SVM + HiSupCon
## 1 1 1
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 1.0, 1.0]_cosine_warm/last.pth" --kernel linear --C 0.01

## 1 2 4
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" --kernel linear --C 0.01

## 4 2 1
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[4.0, 2.0, 1.0]_cosine_warm/last.pth" --kernel linear --C 0.01

## 1 2 5
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 5.0]_cosine_warm/last.pth" --kernel linear --C 0.01

## 1 2 (nAug=1)
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth" --kernel linear --C 0.01

## 1 4 (nAug=1)
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" --kernel linear --C 0.01

## 1 8 (nAug=1)
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 8.0]_cosine_warm/last.pth" --kernel linear --C 0.01

###############

# Consistency + SupCon
cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

# Consistency + HiSupCon
## 1 1 1
cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 1.0, 1.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

## 1 2 4
cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

## 4 2 1
cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[4.0, 2.0, 1.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

## 1 2 5
cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 5.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

## 1 2 (nAug=1)
cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

## 1 4 (nAug=1)
cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

## 1 8 (nAug=1)
cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_consistency_finetune.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 8.0]_cosine_warm/last.pth" --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

##############################################################################

# Train ensemble
srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm_ensemble.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" \
  --model_path2 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth" \
  --kernel linear --C 0.01

srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm_ensemble_consistency.py --batch_size 5000 \
  --model_svm_path1 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" \
  --model_svm_path2 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth" \
  --model_consistency_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth" \
  --C 0.01 --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

# 2 GPUs
srun -n1 -G a100-40g:2 -c 20 --mem 100g -- python run_svm_ensemble_consistency.py --batch_size 10000 \
  --model_svm_path1 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" \
  --model_svm_path2 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth" \
  --model_consistency_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth" \
  --C 0.01 --freeze_epochs 8 --freeze_lr 0.01 --unfreeze_epochs 8 --unfreeze_lr 0.001

srun -n1 -G a100-40g:4 -c 20 --mem 100g -- python run_svm_ensemble.py --batch_size 20000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth"   --model_path2 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth"   --kernel linear --C 0.01 --alpha 0.001 --N 100000 --skip 1

srun -n1 -G a100-40g:4 -c 20 --mem 100g -- python run_svm_ensemble.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth"   --model_path2 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth"   --kernel linear --C 0.01

srun -n1 -G a100-40g:4 -c 20 --mem 100g -- python run_svm_ensemble.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth"   --model_path2 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth"   --kernel linear --C 0.01


srun -n1 -G a100-40g:2 -c 20 --mem 100g -- python run_svm_ensemble.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth" --kernel linear --C 0.01

srun -n1 -G a100-40g:1 -c 20 --mem 100g -- python run_ensemble.py --batch_size 5000 --model_path /scratch/mehrshad_mirmohammadi/Project/smoothing-ensemble-models/models-cifar10/consistency/resnet110/0.25/checkpoint-1000.pth.tar
srun -n1 -G a100-40g:4 -c 20 --mem 100g -- python run_ensemble.py --batch_size 20000 --model_path /scratch/mehrshad_mirmohammadi/Project/smoothing-ensemble-models/models-cifar10/consistency/resnet110/0.25/checkpoint-1000.pth.tar --n_ensembles 4

srun -n1 -G a100-40g:2 -c 20 --mem 100g -- python run_ensemble.py --batch_size 10000 --model_path /scratch/mehrshad_mirmohammadi/Project/smoothing-ensemble-models/models-cifar10/cohen/resnet110/0.25/checkpoint-7000.pth.tar

# Train svm
srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_svm.py --batch_size 5000 --model_path /scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth --kernel linear


##############################################################################
srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python code/certify.py cifar10 logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet50/3490/checkpoint.pth.tar 0.25 logs/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25/cifar_resnet50/3490/certification.txt --batch=5000 --N=10000 --skip=10 --alpha=0.01

srun -n1 -G a100-40g:2 -c 24 --mem 160g -- python main_supcon_hierarchy.py --batch_size 512  --learning_rate 0.5 --temp 0.1 --cosine --n_aug_view 2 --n_noise_view 2 --alpha 3.0 2.0 1.0
srun -n1 -G a100-40g:2 -c 20 --mem 100g -- python main_supcon_hierarchy.py --batch_size 512   --learning_rate 0.5   --temp 0.1   --cosine   --trial Sequential-BenchMarking   --alpha 1 1 1

source ../anaconda3/bin/activate && cd SelfSupForRanSmooth/notebooks/TrainAndCertifyWithFewerSamples/ && clear

cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_svm.py --batch_size 5000 --model_path /scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth --kernel linear --C 0.001

srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_svm_FineTune.py --batch_size 5000 --model_path /scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth   --freeze_epochs 8   --freeze_lr 0.01   --unfreeze_epochs 8   --unfreeze_lr 0.001

srun -n1 -G a100-40g:1 -c 12 --mem 80g -- python run_hinge.py --batch_size 5000 --model_path /scratch/mehrshad_mirmohammadi/Project/SupContrast/save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential_cosine_warm/last.pth   --freeze_epochs 8   --freeze_lr 0.01   --unfreeze_epochs 8   --unfreeze_lr 0.001



################ Train Consistency + HieSupCon ################

source ../anaconda3/bin/activate && cd SupContrast && clear

# Train Hierarchical Contrastive Learning
srun -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy_with_Consistency.py --batch_size 512 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 2 \
  --consistency_ratio 0.5


################ Train Two Level Only pull + SupCon ################

srun -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy_pullOnly.py --batch_size 512 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 2

################ Train Two Level with Wide ResNet ################

srun -w gcp-us-1 -n1 -G a100-40g:2 -c 16 --mem 100g -- python main_supcon_hierarchy.py --batch_size 512 --n_aug_view 1 \
  --num_workers 16 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --alpha 1 0 2 \
  --model wide_resnet50_2


################ SVM + HieSupCon FULL ################

# ## 1 2 (nAug=1)
# cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a100-40g:1 -c 10 --mem 50g -- python run_svm.py --batch_size 5000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth" --kernel linear --C 0.01

## 1 4 (nAug=1)
cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -n1 -G a6000:1 -c 10 --mem 50g -- python run_svm.py --batch_size 8000 --model_path "/home/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth" --kernel linear --C 0.01 --alpha 0.001 --N 100000 --skip 1


srun -n1 -G a100-40g:2 -c 20 --mem 100g -- python run_svm.py --batch_size 8000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth" --kernel linear --C 0.01 --alpha 0.001 --N 100000 --skip 1

# srun -n1 -G a100-40g:4 -c 20 --mem 100g -- python run_svm_ensemble.py --batch_size 20000 --model_path "/home/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_2_nNoise_2_alpha_[1.0, 2.0, 4.0]_cosine_warm/last.pth"   --model_path2 "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_Sequential-BenchMarking_nAug_1_nNoise_2_alpha_[1.0, 2.0, 2.0]_cosine_warm/last.pth"   --kernel linear --C 0.01 --alpha 0.001 --N 100000 --skip 1

################ SVM stuff ################

cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -w gcp-eu-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_svm.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/OnlyPullOtherThanClass_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_2_nNoise_2_alpha_[1.0, 0.0, 2.0]_cosine_warm/last.pth" --kernel linear --C 0.01

cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -w gcp-eu-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_svm.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/OnlyPullOtherThanClass_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_2_nNoise_2_alpha_[1.0, 0.0, 8.0]_cosine_warm/last.pth" --kernel linear --C 0.01

cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -w gcp-eu-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_svm.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/OnlyPullOtherThanClass_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_2_nNoise_2_alpha_[2.0, 0.0, 1.0]_cosine_warm/last.pth" --kernel linear --C 0.01

cd SelfSupForRanSmooth/notebooks/SVM/ && clear && srun -w gcp-eu-1 -n1 -G a100-40g:2 -c 16 --mem 50g -- python run_svm.py --batch_size 10000 --model_path "/scratch/mehrshad_mirmohammadi/Project/SupContrast/save/HiSupCon/cifar10_models/cifar10_wide_resnet50_2_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_nAug_1_nNoise_2_alpha_[1.0, 0.0, 2.0]_cosine_warm/last.pth" --kernel linear --C 0.01

####

