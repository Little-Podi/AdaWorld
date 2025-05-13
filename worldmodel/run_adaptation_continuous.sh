torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_adapt.py \
    --base configs/training/adaworld_adapt_continuous_action.yaml \
    --finetune ckpts/adaworld.safetensors \
    --num_nodes 1 \
    --n_devices 8 \
    2>&1 | tee output_adapt_continuous.log