# Learning Adaptable World Models with Latent Actions

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)

[[Project Page](https://adaptable-world-model.github.io)] [[Technical Report](https://arxiv.org/abs/2503.xxxxx)]

> [Shenyuan Gao](https://github.com/Little-Podi), [Siyuan Zhou](https://scholar.google.com/citations?user=WjUmtm0AAAAJ), [Yilun Du](https://yilundu.github.io), [Jun Zhang](https://eejzhang.people.ust.hk), [Chuang Gan](https://people.csail.mit.edu/ganchuang)

<hr style="border: 2px solid gray;"></hr>

**TL;DR:** *AdaWorld is a highly adaptable world model trained with continuous latent actions, enabling efficient action transfer, world model adaptation, and visual planning.*

- Efficient action transfer (source video &rarr; target scene)

<div id="top" align="center">
<p align="center">
<img src="assets/transfer.gif" width="1000px" >
</p>
</div>

- Effective agent planning (action-agnostic vs. AdaWorld)

<div id="top" align="center">
<p align="center">
<img src="assets/planning.gif" width="1000px" >
</p>
</div>

![](assets/teaser.png)

We introduce latent actions as a unified condition for action-aware pretraining from videos. Our world model, dubbed AdaWorld, can readily transfer actions across contexts without training. By initializing action embeddings with corresponding latent actions, AdaWorld can also be adapted into specialized world models through limited interactions and finetuning steps.

## TODO List

- [ ] Provide detailed instructions to get started with this project.
- [ ] Upload the source of our dataset.
- [ ] Release model weights.

## Acknowledgement

Our idea is implemented base on [Vista](https://github.com/OpenDriveLab/Vista) and [Jafar](https://github.com/flairox/jafar). Thanks for their great open-source work!

## Citation
If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.
```
@article{gao2025adaworld,
  title={AdaWorld: Learning Adaptable World Models with Latent Actions}, 
  author={Gao, Shenyuan and Zhou, Siyuan and Du, Yilun and Zhang, Jun and Gan, Chuang},
  journal={arXiv preprint arXiv:2503.xxxxx},
  year={2025}
}
```

## Contact

If you have any questions or concerns, feel free to contact me through email (sygao@connect.ust.hk). Suggestions and collaborations are also highly welcome!
