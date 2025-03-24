python main.py test \
    --ckpt_path checkpoints/lam/epoch=1.ckpt \
    --config config/lam.yaml \
    2>&1 | tee output_test.log