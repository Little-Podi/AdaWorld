# Trouble Shooting

1. `TypeError: write_frames() got an unexpected keyword argument 'audio_path'` when sampling videos.

   - This is caused by the low version of `imageio-ffmpeg`.
   - Upgrade it to fix: `pip install --upgrade imageio-ffmpeg==0.4.8`.

2. `IndexError: list index out of range` when training latent action autoencoder.

   - You may forget to align `nnodes` and `nproc_per_node` in `lam/train.sh` with your GPU setup.
   - Please check `lam/train.sh` and modify them accordingly.

3. The number of training iterations does not match when training latent action autoencoder.

   - You may forget to align `num_nodes` and `devices` in `lam/config/lam.yaml` with your GPU setup.
   - Please check `lam/config/lam.yaml` and modify them accordingly.

---

<= Previous: [[Visual Planning](https://github.com/Little-Podi/AdaWorld/blob/main/docs/PLANNING.md)]