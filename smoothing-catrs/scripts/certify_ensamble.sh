agg_scheme=0

CUDA_VISIBLE_DEVICES=7 python code/certify_ensemble_gpu.py \
    cifar10 \
    0.25 \
    test/certify/cifar10/ensamble/all/$agg_scheme/noise_0.25.tsv \
    /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_catrs.pt \
    /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_macer_2.pt \
    /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_cohen.pt  \
    /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_consistency.pt \
    /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_salman.pt \
    /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_smix_0.5_4_m0.pt \
    /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_stab.pt \
    /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_supcon_2l_2.pt \
    --alpha 0.001 \
    --N 100000 \
    --skip 1 \
    --combinations 4 \
    --comb-dict 01234569 \
    --aggregation-scheme $agg_scheme \
    --softmax-idx 7 \
    --softmax-temp 1 1.5 1.6 1.7 \
    --batch 500

    # --combinations-to-analyze 0123 013 023 \
    # /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_supcon_2l.pt \
      
    # /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_supcon_3l.pt \


    # /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.25/predictions_100000_macer.pt \
    # /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_macer.pt \

# CUDA_VISIBLE_DEVICES=6 python code/certify_ensemble_gpu.py \
#     cifar10 \
#     0.5 \
#     test/certify/cifar10/ensamble/all/$agg_scheme/0.5/noise_0.5.tsv \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_catrs.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_macer_2.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_cohen.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_consistency.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_salman.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_smix_1.0_4_m1.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_stab.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/0.5/predictions_100000_supcon_2l.pt \
#     --alpha 0.001 \
#     --N 100000 \
#     --skip 1 \
#     --combinations 4 \
#     --aggregation-scheme $agg_scheme \
#     --batch 500

# #     # --comb-dict 0457 \


# agg_scheme=0
# CUDA_VISIBLE_DEVICES=7 python code/certify_ensemble_gpu.py \
#     cifar10 \
#     1.0 \
#     test/certify/cifar10/ensamble/all/$agg_scheme/1.0/noise_1.0.tsv \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/1.0/predictions_100000_catrs.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/1.0/predictions_100000_macer_deferred200_2.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/1.0/predictions_100000_cohen.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/1.0/predictions_100000_salman_2.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/1.0/predictions_100000_smix_2.0_4_m1.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/1.0/predictions_100000_stab.pt \
#     /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/1.0/predictions_100000_supcon_2l.pt \
#     --alpha 0.001 \
#     --N 100000 \
#     --skip 1 \
#     --combinations 4 \
#     --comb-dict 0124567 \
#     --aggregation-scheme $agg_scheme \
#     --batch 500

    # /mnt/disks/rs-cert-data/predictions/cifar10/gaussian/1.0/predictions_100000_consistency.pt \
