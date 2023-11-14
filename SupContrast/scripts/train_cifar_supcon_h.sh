CUDA_VISIBLE_DEVICES=0 python main_supcon_hierarchy.py \
    --model resnet20_slim \
    --batch_size 512 --learning_rate 0.5 \
    --temp 0.1 --cosine \
    --trial Sequential-BenchMarking \
    --alpha 1 2 2 --n_aug_view 1 --sigma 0.25
