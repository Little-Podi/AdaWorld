# Trouble Shooting

1. `TypeError: write_frames() got an unexpected keyword argument 'audio_path'` when saving videos.

   - This is caused by the low version of `imageio-ffmpeg`.
   - Upgrade it to fix: `pip install --upgrade imageio-ffmpeg==0.4.8`.

2. `IndexError: list index out of range` when training latent action autoencoder.

   - You may forget to align `nnodes` and `nproc_per_node` in `lam/train.sh` with your GPU setup.
   - Please check `lam/train.sh` and modify them accordingly.

3. The number of training iterations does not match when training latent action autoencoder.

   - You may forget to align `num_nodes` and `devices` in `lam/config/lam.yaml` with your GPU setup.
   - Please check `lam/config/lam.yaml` and modify them accordingly.

4. Get stuck at loading FrozenCLIPEmbedder when training the world model.

    - This is likely due to a network failure.
    - Download the model from [Hugging Face](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) in advance and set `version` of FrozenOpenCLIPImageEmbedder in `worldmodel/vwm/modules/encoders/modules.py` to the path of `open_clip_pytorch_model.bin`.

---

<= Previous: [[Visual Planning](https://github.com/Little-Podi/AdaWorld/blob/main/docs/PLANNING.md)]