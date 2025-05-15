# World Model Adaptation

In our paper, we borrow the mini DMLab and Minecraft datasets from [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing) for world model adaptation experiments.

### For environments with discrete action spaces

1. Split the collected samples into short video clips consisting of two frames, and organize them into different folders according to action indices. Take the Minecraft with 3 action options as an example:
   ```
   data/
    |--minecraft/
        |--action_0/
        |   |--00000.mp4
        |   |...
        |--action_1/
        |   |--00000.mp4
        |   |...
        |--action_2/
            |--00000.mp4
            |...
   ```
2. Go through each action folder to infer the latent actions using the pretrained latent action encoder. This can be done by running `lam/test.sh` and setting `batch_size` in `lam/config/lam.yaml` to 1.
3. Uncomment the `on_test_epoch_end` function in `lam/lam/model.py` to save the inferred latent actions as `latent_action_stats.pt`.
4. In MultiSourceSamplerDataset, replace VideoDataset with VideoDatasetDiscreteActionSpace. Please check the parameter inputs and rename all paths if necessary.
5. Use the averaged latent actions as the action embeddings for the discrete action codebook of ActionBook in `worldmodel/vwm/modules/encoders/modules.py`. An example is provided in `__init__`.
6. Run `worldmodel/run_adaptation_discrete.sh`.

### For environments with continuous action spaces

1. Split the collected samples into short video clips consisting of two frames, and save the action values using the same file name. Take the nuScenes with a two-dimensional action displacement as an example:
   ```
    data/
    |--nuscenes/
        |--00000.mp4
        |--00000.txt
        |--00001.mp4
        |--00001.txt
        |...
   ```
   The TXT files store a list that contains the displacement [x,y] of each transition.
2. Go through all video clips to infer their latent actions using the pretrained latent action encoder. This can be done by running `lam/test.sh` and setting `batch_size` in `lam/config/lam.yaml` to 1.
3. Uncomment the `on_test_epoch_end` function in `lam/lam/model.py` to save the inferred latent actions as `latent_action_stats.pt`.
4. In MultiSourceSamplerDataset, replace VideoDataset with VideoDatasetContinuousActionSpace. Please check the parameter inputs and rename all paths if necessary.
5. Convert the ground truth of all actions to `raw_action_inputs.pt`, ensuring it corresponds to the order of `latent_action_stats.pt`.
6. Use `worldmodel/fast_init_mlp.py` to optimize the initialization weights `mlp_init_weights.pth` for ActionMLP in `worldmodel/vwm/modules/encoders/modules.py`. An example is provided in `__init__`.
7. Run `worldmodel/run_adaptation_continuous.sh`.

To visualize the UMAP projection of latent actions in our paper, please refer to [UMAP](https://github.com/lmcinnes/umap) and set `n_neighbors` and `min_dist` to 15 and 0.5, respectively.

---

<= Previous: [[Action Transfer](https://github.com/Little-Podi/AdaWorld/blob/main/docs/TRANSFER.md)]

=> Next: [[Visual Planning](https://github.com/Little-Podi/AdaWorld/blob/main/docs/PLANNING.md)]