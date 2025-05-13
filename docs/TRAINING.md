# Training

## Latent Action Autoencoder

To train the latent action autoencoder (suppose you train on 1 node with 8 GPUs):

```shell
cd lam
train.sh
```

- Remember to modify `num_nodes` and `devices` in `lam/config/lam.yaml` accordingly if you have a different GPU setup.
- Remember to set `max_epochs` or stop the training when you think the training is long enough.

## World Model Pretraining

### Preparation

- Download the pretrained Stable Video Diffusion checkpoint `svd.safetensors` from [Hugging Face](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid).
- Reset `default_ckpt` in `worldmodel/train.py` with the path of `svd.safetensors`.
- Reset `ckpt_path` in `worldmodel/configs/training/adaworld.yaml` with the last checkpoint path of the latent action autoencoder.

To pretrain the autoregressive world model (suppose you train on 1 node with 8 GPUs):

```shell
cd worldmodel
run_train.sh
```

---

<= Previous: [[Installation](https://github.com/Little-Podi/AdaWorld/blob/main/docs/INSTALLATION.md)]

=> Next: [[Action Transfer](https://github.com/Little-Podi/AdaWorld/blob/main/docs/TRANSFER.md)]