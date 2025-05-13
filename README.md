# Learning Adaptable World Models with Latent Actions

[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/Little-Podi/AdaWorld)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)

🎬 [[Project Page](https://adaptable-world-model.github.io)], 📜 [[Technical Report](https://arxiv.org/abs/2503.18938)], 🤗 [[Model Weights](https://huggingface.co/Little-Podi/AdaWorld)]

[Shenyuan Gao](https://github.com/Little-Podi), [Siyuan Zhou](https://rainbow979.github.io), [Yilun Du](https://yilundu.github.io), [Jun Zhang](https://eejzhang.people.ust.hk), [Chuang Gan](https://people.csail.mit.edu/ganchuang)

<hr style="border: 2px solid gray;"></hr>

**TL;DR:** *We pretrain AdaWorld with continuous latent actions from thousands of environments, making it highly efficient for action transfer, world model adaptation, and visual planning, even with minimal interactions and finetuning steps.*

- Action transfer (source video &rarr; target scene)

<div id="top" align="center">
<p align="center">
<img src="assets/transfer.gif" width="1000px" >
</p>
</div>

- Visual planning (action-agnostic vs. AdaWorld)

<div id="top" align="center">
<p align="center">
<img src="assets/planning.gif" width="1000px" >
</p>
</div>

<hr style="border: 2px solid gray;"></hr>

![](assets/teaser.png)

We introduce latent actions as a unified condition for action-aware pretraining from videos. AdaWorld can readily transfer actions across contexts without training. By initializing the control interface with the corresponding latent actions, AdaWorld can also be adapted into specialized world models efficiently and achieve significantly better planning results.

## 🕹️ Getting Started

- [Installation](https://github.com/Little-Podi/AdaWorld/blob/main/docs/INSTALLATION.md)
- [Training](https://github.com/Little-Podi/AdaWorld/blob/main/docs/TRAINING.md)
- [Action Transfer](https://github.com/Little-Podi/AdaWorld/blob/main/docs/TRANSFER.md)
- [World Model Adaptation](https://github.com/Little-Podi/AdaWorld/blob/main/docs/ADAPTATION.md)
- [Visual Planning](https://github.com/Little-Podi/AdaWorld/blob/main/docs/PLANNING.md)
- [Trouble Shooting](https://github.com/Little-Podi/AdaWorld/blob/main/docs/ISSUES.md)

## ❤️ Acknowledgement

Our idea is implemented based on [Vista](https://github.com/OpenDriveLab/Vista) and [Jafar](https://github.com/flairox/jafar). Thanks for their great open-source work!

## ⭐ Citation
If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.
```
@article{gao2025adaworld,
  title={AdaWorld: Learning Adaptable World Models with Latent Actions}, 
  author={Gao, Shenyuan and Zhou, Siyuan and Du, Yilun and Zhang, Jun and Gan, Chuang},
  journal={arXiv preprint arXiv:2503.18938},
  year={2025}
}
```

## 📢 Contact

If you have any questions or comments, feel free to contact me through email (sygao@connect.ust.hk). Suggestions and collaborations are also highly welcome!
