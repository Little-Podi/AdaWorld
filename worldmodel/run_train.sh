torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
    --base configs/training/adaworld.yaml \
    --num_nodes 1 \
    --n_devices 8 \
    2>&1 | tee output_train.log