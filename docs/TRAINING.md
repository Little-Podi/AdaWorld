# Training

### Latent Action Autoencoder

To train the latent action autoencoder (suppose you train on 1 node with 8 GPUs):
```shell
cd lam
train.sh
```

> [!NOTE]
> The checkpoint of our latent action autoencoder can be found at [Hugging Face](https://huggingface.co/Little-Podi/AdaWorld/blob/main/lam.ckpt).

### World Model Pretraining

1. Download the pretrained Stable Video Diffusion checkpoint `svd.safetensors` from [Hugging Face](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid).
2. Reset `default_ckpt` in `worldmodel/train.py` with the path of `svd.safetensors`.
3. Reset `ckpt_path` in `worldmodel/configs/training/adaworld.yaml` with the last checkpoint path of the latent action autoencoder.

To pretrain the autoregressive world model (suppose you train on 1 node with 8 GPUs):
```shell
cd worldmodel
run_train.sh
```

After training, you can convert the checkpoint to safetensors format using `worldmodel/bin_to_st.py` and do inference.

> [!NOTE]
> The pretrained AdaWorld can be found at [Hugging Face](https://huggingface.co/Little-Podi/AdaWorld/blob/main/adaworld.safetensors).

> [!TIP]
> Remember to modify `num_nodes` and `devices` in `lam/config/lam.yaml` accordingly if you have a different GPU setup.
> 
> Remember to set `max_epochs` or stop the training when you think the training is long enough.

---

<= Previous: [[Installation](https://github.com/Little-Podi/AdaWorld/blob/main/docs/INSTALLATION.md)]

=> Next: [[Action Transfer](https://github.com/Little-Podi/AdaWorld/blob/main/docs/TRANSFER.md)]