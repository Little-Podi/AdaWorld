# Installation

## Environment

We use conda to manage the environment. The installation is tested with CUDA 11.8 and NVIDIA A100 (80 GB).

```shell
conda create -n adaworld python=3.10 -y
conda activate adaworld
pip install -r requirements.txt
```

## Dataset

In this instruction, we provide our automated generation process on [Procgen](https://github.com/openai/procgen) and [Gym Retro](https://github.com/openai/retro) as examples.

For Gym Retro, we recommend installing an independent conda environment for automated data generation following this [guide](https://retro.readthedocs.io/en/latest/getting_started.html). After installation, please manually import [ROMs](https://archive.org/details/No-Intro-Collection_2016-01-03_Fixed) for interaction. To obtain all game environments listed in the paper, you may need to search for the missing ROMs by [name](https://github.com/openai/retro/tree/master/retro/data/stable) on the web and import them individually. We also have 12 extra games imported from [Stable-Retro](https://github.com/Farama-Foundation/stable-retro) to further enrich our data corpus.

Run the following scripts to sample videos using a biased random agent (may take a while). You can customize the number and length of the videos by modifying `num_logs` and `timeout`.

```shell
python sample_procgen.py
python sample_retro.py
python sample_stableretro.py
```

The organized data directory should look like:

```
data/
|--procgen/
|   |--bigfish/
|   |   |--test/
|   |   |   |--00000.mp4
|   |   |   |...
|   |   |--train/
|   |       |--00000.mp4
|   |       |...
|   |--bossfight/
|   |   |--test/
|   |   |   |--00000.mp4
|   |   |   |...
|   |   |--train/
|   |       |--00000.mp4
|   |       |...
|   |...
|--retro/
    |--3NinjasKickBack-Genesis/
    |   |--test/
    |   |   |--00000.mp4
    |   |   |...
    |   |--train/
    |       |--00000.mp4
    |       |...
    |--8Eyes-Nes/
    |   |--test/
    |   |   |--00000.mp4
    |   |   |...
    |   |--train/
    |       |--00000.mp4
    |       |...
    |...
```

You can also download and compile other datasets we used:

- [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment)
  - We extract as many video sequences as possible regardless of their viewpoint.
  - Please find `download_open_x.sh` and `process_rtx.py` to see how we extract raw videos.
- [Ego4D](https://github.com/facebookresearch/Ego4d)
- [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
- [MiraData](https://github.com/mira-space/MiraData)
  - We only use the first 8K videos (3D rendered games and city walking tours) in the list.
  - You can download more if you want and have sufficient storage.

or those not used (but we have tried):

- [Panda-70M](https://github.com/snap-research/Panda-70M)
- [Ego-Exo4D](https://github.com/facebookresearch/Ego4d)
- [EPIC-KITCHENS](https://github.com/epic-kitchens/epic-kitchens-download-scripts)
- [OGameData](https://github.com/GameGen-X/GameGen-X)

as folders of `.mp4`/`.webm` files.

---

=> Next: [[Training](https://github.com/Little-Podi/AdaWorld/blob/main/docs/TRAINING.md)]