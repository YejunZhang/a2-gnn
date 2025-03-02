

python -m a2gnn_train.train_matcher --gpus 0 --batch 16 -lr 0.001 \
    --max_epochs 50 --matcher_class 'OTMatcherCls' --share_kp2d_enc \
    --dataset 'megadepth' --train_split 'train' --val_split 'val' \
    --outlier_rate 0.5 0.5  --topk 1 --npts 100 1024 \
    --p2d_type 'sift' --p3d_type 'bvs' \
    --inls2d_thres 0.001 --rpthres 0.01 