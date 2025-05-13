# Visual Planning

After efficient adaptation, AdaWorld can be controlled directly by raw action inputs, which enables visual planning. Please refer to the setup of [iVideoGPT](https://github.com/thuml/iVideoGPT/tree/main/vp) to compile our world model and perform robot planning tasks on the [VP2](https://github.com/s-tian/vp2) benchmark. You may need to batchify the sampling, reduce the denoising steps (to 3), and turn off classifier-free guidance (use IdentityGuider), otherwise it could take days.

> [!TIP]
> Use [Hugging Face](https://huggingface.co/datasets/s-tian/VP2) to download the VP2 dataset. It will be much faster than the older links.
> For Robosuite, we only require `5k_slice_rendered_256.hdf5`. For RoboDesk, you can use the scripted policies to generate the training trajectories locally if the download feels too slow.

---

<= Previous: [[World Model Adaptation](https://github.com/Little-Podi/AdaWorld/blob/main/docs/ADAPTATION.md)]

=> Next: [[Trouble Shooting](https://github.com/Little-Podi/AdaWorld/blob/main/docs/ISSUES.md)]