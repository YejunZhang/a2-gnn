#!/bin/bash
#SBATCH --job-name=gomatch
#SBATCH --account=project_2002051
#SBATCH --partition=gpu
#SBATCH --time=2:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:v100:1

# Eval On Megadepth  superpoint  r2d2

python -m gomatch_eval.benchmark  --root .  --ckpt /home/zhangy36/code/3dv/gomatch_yejun/outputs/a2best.ckpt\
    --splits 'test'  \
    --odir 'outputs/3dv++/eval/a2best1' \
    --dataset 'megadepth' --covis_k_nums 10 \
    --p2d_type 'sift' 
    # --merge_before_match

# Eval On Cambridge Landmarks
# python -m gomatch_eval.benchmark  --root .  --ckpt /scratch/project_2002051/zhangyejun/thesis/gomatch_yejun/outputs/gomatch_annular_cl_coord_fusion_ang++_color/megadepth/train_val/or0.5-0.5top1sift_bvs100-1024inls0.001/gomatchbvs.OTMatcherCls.share2d.scs_rpthres0.01/batch16_lr0.001/version0/checkpoints/best.ckpt \
#     --splits  'all' \
#     --odir 'outputs/eval/cambridge/StMarysChurch' \
#     --dataset 'cambridge' --covis_k_nums 10  \
#     --p2d_type 'superpoint'

# 'chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs' \
# Eval On 7 Scenes 
# python -m gomatch_eval.benchmark  --root .  --ckpt /scratch/project_2002051/zhangyejun/thesis/gomatch_yejun/outputs/3dv_release/best2/megadepth/train_val/or0.5-0.5top1sift_bvs100-1024inls0.001/gomatchbvs.OTMatcherCls.share2d.scs_rpthres0.01/batch16_lr0.001/version0/checkpoints/best.ckpt \
#     --splits 'chess' \
#     --odir 'outputs/eval/7scenes/best2_chess' \
#     --dataset '7scenes_sift_v2' --covis_k_nums 10  \
#     --p2d_type 'sift'
# python -m gomatch_eval.benchmark  --root .  --ckpt /scratch/project_2002051/zhangyejun/thesis/gomatch_yejun/outputs/3dv_release/best2/megadepth/train_val/or0.5-0.5top1sift_bvs100-1024inls0.001/gomatchbvs.OTMatcherCls.share2d.scs_rpthres0.01/batch16_lr0.001/version0/checkpoints/best.ckpt \
#     --splits 'stairs' \
#     --odir 'outputs/eval/7scenes/sp_stairs' \
#     --dataset '7scenes_superpoint_v2' --covis_k_nums 10  \
#     --p2d_type 'superpoint'