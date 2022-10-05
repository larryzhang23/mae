#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -W ignore -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_small_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /data/jiarui/data/imagenet/2012 \
    --num_workers 4 \
    --wandb \
    --resume ./output_dir/checkpoint-60.pth


#CUDA_VISIBLE_DEVICES=0,3,5,6 python -W ignore main_pretrain.py \
#    --batch_size 64 \
#    --model mae_vit_small_patch16 \
#    --norm_pix_loss \
#    --mask_ratio 0.75 \
#    --epochs 800 \
#    --warmup_epochs 40 \
#    --blr 1.5e-4 --weight_decay 0.05 \
#    --data_path /data/jiarui/data/imagenet/2012 \
#    --num_workers 4 \
#    --wandb