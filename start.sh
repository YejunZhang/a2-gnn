#!/bin/bash
#SBATCH --job-name=gomatch
#SBATCH --account=project_2002051
#SBATCH --partition=gpu
#SBATCH --time=47:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --gres=gpu:v100:1


python -m gomatch_train.train_matcher --gpus 0 --batch 16 -lr 0.001 \
    --max_epochs 50 --matcher_class 'OTMatcherCls' --share_kp2d_enc \
    --dataset 'megadepth' --train_split 'train' --val_split 'val' \
    --outlier_rate 0.5 0.5  --topk 1 --npts 100 1024 \
    --p2d_type 'sift' --p3d_type 'bvs' \
    --inls2d_thres 0.001 --rpthres 0.01 --prefix 'gomatchbvs' \
    -o 'outputs/supplymentary' --num_workers 4 
    # --att_layers self1 cross self1
# nei_12_group_3
# mix_up_fix scannet megadepth
# change the detector: p2d_type 'sift', 'superpoint', 'r2d2'