
python -m a2gnn_eval.benchmark  --root .  --ckpt <model path> \
    --splits 'test'  \
    --odir  <output path> \
    --dataset 'megadepth' --covis_k_nums 10 \
    --p2d_type 'sift' 
