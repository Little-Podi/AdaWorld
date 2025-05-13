import math
from os import listdir, path
from random import choices, randint
from typing import Dict, List, Tuple

import ast
import cv2 as cv
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class VideoDataset(Dataset):
    def __init__(
            self,
            split_path: str,
            padding: str = "repeat",
            randomize: bool = False,
            resolution: int = 256,
            n_context_frames: int = 5,
            output_format: str = "t c h w",
            color_aug: bool = True
    ):
        super(VideoDataset, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.resolution = resolution
        self.n_context_frames = n_context_frames
        self.output_format = output_format
        self.color_aug = color_aug

        # Get all the file path based on the split path
        self.file_names = []
        for file_name in listdir(split_path):
            if file_name.endswith(".mp4") or file_name.endswith(".webm"):
                self.file_names.append(path.join(split_path, file_name))

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        while True:
            try:
                image_seq = self.load_video_slice(
                    video_path,
                    self.n_context_frames + 1,
                    None if self.randomize else 0
                )
                return self.build_data_dict(image_seq)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
            self,
            video_path: str,
            num_frames: int,
            start_frame: int = None,
            frame_skip: int = 1
    ) -> List:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if "retro" in video_path:
            frame_skip = 4
        elif "procgen" not in video_path and "ssv2" not in video_path and "mira" not in video_path:
            frame_skip = 2
        num_frames = num_frames * frame_skip

        start_frame = randint(0, max(0, total_frames - num_frames)) if start_frame is None else start_frame
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # Frame was successfully read, parse it
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                # Reach the end of video, deal with padding and return
                if self.padding == "none":
                    pass
                elif self.padding == "repeat":
                    frames.extend([frames[-1]] * (num_frames - len(frames)))
                elif self.padding == "zero":
                    frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                elif self.padding == "random":
                    frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                else:
                    raise ValueError(f"Invalid padding type: {self.padding}")
                break
        cap.release()

        video = torch.stack(frames[::frame_skip]) / 255.0

        # Crop the video to be square
        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> t c h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"t c h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")

        if self.color_aug:
            # Brightness jitter
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return [frame for frame in video]

    def build_data_dict(self, image_seq: List) -> Dict:
        context_len = choices(range(1, self.n_context_frames + 1))[0]
        filled_seq = [torch.zeros_like(image_seq[0])] * (self.n_context_frames - context_len) + image_seq
        next_frames = torch.Tensor(filled_seq[self.n_context_frames])
        prev_frames = torch.stack(filled_seq[:self.n_context_frames])
        lam_inputs = torch.stack(filled_seq[self.n_context_frames - 1:self.n_context_frames + 1])
        context_len = torch.Tensor([context_len])
        next_frames = next_frames * 2.0 - 1.0
        prev_frames = prev_frames * 2.0 - 1.0
        context_aug = torch.Tensor(choices(range(8))) / 10
        img_seq = torch.cat([prev_frames, next_frames[None]])
        data_dict = {
            "img_seq": img_seq,  # (T, 3, 256, 256)
            "cond_frames_without_noise": prev_frames[-1],
            "cond_frames": prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]),
            "lam_inputs": lam_inputs,  # (2, 3, 256, 256)
            "context_len": context_len,  # (1,)
            "context_aug": context_aug  # (1,)
        }
        return data_dict


class VideoDatasetDiscreteActionSpace(Dataset):
    def __init__(
            self,
            split_path: str,
            randomize: bool = False,
            resolution: int = 256,
            n_context_frames: int = 5,
            output_format: str = "t c h w",
            color_aug: bool = True
    ):
        super(VideoDatasetDiscreteActionSpace, self).__init__()
        self.randomize = randomize
        self.resolution = resolution
        self.n_context_frames = n_context_frames
        self.output_format = output_format
        self.color_aug = color_aug

        # Get all the file path based on the split path
        self.file_names = []
        for file_name in listdir(split_path):
            if file_name.endswith(".mp4") or file_name.endswith(".webm"):
                self.file_names.append(path.join(split_path, file_name))

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        while True:
            try:
                image_seq, raw_action = self.load_video_slice(
                    video_path,
                    self.n_context_frames + 1
                )
                return self.build_data_dict(image_seq, raw_action)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
            self,
            video_path: str,
            num_frames: int
    ) -> Tuple[List, torch.Tensor]:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        assert num_frames == total_frames

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # Frame was successfully read, parse it
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                # Reach the end of video
                raise NotImplementedError
        cap.release()

        video = torch.stack(frames) / 255.0

        # Crop the video to be square
        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> t c h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"t c h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")

        if self.color_aug:
            # Brightness jitter
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)

        action_anno = video_path.split("action_")[1].split("/")[0]
        raw_action = torch.Tensor([int(action_anno)])
        return [frame for frame in video], raw_action

    def build_data_dict(self, image_seq: List, raw_action: torch.Tensor) -> Dict:
        context_len = choices(range(1, self.n_context_frames + 1))[0]
        image_seq = image_seq[self.n_context_frames - context_len:]
        filled_seq = [torch.zeros_like(image_seq[0])] * (self.n_context_frames - context_len) + image_seq
        next_frames = torch.Tensor(filled_seq[self.n_context_frames])
        prev_frames = torch.stack(filled_seq[:self.n_context_frames])
        context_len = torch.Tensor([context_len])
        next_frames = next_frames * 2.0 - 1.0
        prev_frames = prev_frames * 2.0 - 1.0
        context_aug = torch.Tensor(choices(range(8))) / 10
        img_seq = torch.cat([prev_frames, next_frames[None]])
        data_dict = {
            "img_seq": img_seq,  # (T, 3, 256, 256)
            "cond_frames_without_noise": prev_frames[-1],
            "cond_frames": prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]),
            "context_len": context_len,  # (1,)
            "context_aug": context_aug,  # (1,)
            "raw_action": raw_action
        }
        return data_dict


class VideoDatasetContinuousActionSpace(Dataset):
    def __init__(
            self,
            split_path: str,
            randomize: bool = False,
            resolution: int = 256,
            n_context_frames: int = 5,
            output_format: str = "t c h w",
            color_aug: bool = True
    ):
        super(VideoDatasetContinuousActionSpace, self).__init__()
        self.randomize = randomize
        self.resolution = resolution
        self.n_context_frames = n_context_frames
        self.output_format = output_format
        self.color_aug = color_aug

        # Get all the file path based on the split path
        self.file_names = []
        self.actions = []
        for file_name in listdir(split_path):
            if file_name.endswith(".mp4") or file_name.endswith(".webm"):
                self.file_names.append(path.join(split_path, file_name))

                action_txt = file_name.replace(".mp4", ".txt")
                action_file = open(path.join(split_path, action_txt), "r")
                actions = action_file.read().splitlines()[0]
                actions = ast.literal_eval(actions)
                self.actions.append(actions)

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        raw_action = self.actions[idx]
        while True:
            try:
                image_seq = self.load_video_slice(
                    video_path,
                    self.n_context_frames + 1
                )
                return self.build_data_dict(image_seq, raw_action)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]
                raw_action = self.actions[idx]

    def load_video_slice(
            self,
            video_path: str,
            num_frames: int
    ) -> List:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        assert num_frames == total_frames

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # Frame was successfully read, parse it
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                # Reach the end of video
                raise NotImplementedError
        cap.release()

        video = torch.stack(frames) / 255.0

        # Crop the video to be square
        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> t c h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"t c h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")

        if self.color_aug:
            # Brightness jitter
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return [frame for frame in video]

    def build_data_dict(self, image_seq: List, raw_action: List) -> Dict:
        context_len = choices(range(1, self.n_context_frames + 1))[0]
        image_seq = image_seq[self.n_context_frames - context_len:]
        filled_seq = [torch.zeros_like(image_seq[0])] * (self.n_context_frames - context_len) + image_seq
        next_frames = torch.Tensor(filled_seq[self.n_context_frames])
        prev_frames = torch.stack(filled_seq[:self.n_context_frames])
        context_len = torch.Tensor([context_len])
        next_frames = next_frames * 2.0 - 1.0
        prev_frames = prev_frames * 2.0 - 1.0
        context_aug = torch.Tensor(choices(range(8))) / 10
        img_seq = torch.cat([prev_frames, next_frames[None]])
        data_dict = {
            "img_seq": img_seq,  # (T, 3, 256, 256)
            "cond_frames_without_noise": prev_frames[-1],
            "cond_frames": prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]),
            "context_len": context_len,  # (1,)
            "context_aug": context_aug,  # (1,)
            "raw_action": torch.Tensor(raw_action)
        }
        return data_dict


class MultiSourceSamplerDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            env_source: str = "libero",
            split: str = "train",
            samples_per_epoch: int = 60000,
            sampling_strategy: str = "pi",
            **kwargs
    ):
        self.samples_per_epoch = samples_per_epoch

        # Create all subsets
        folders = []
        if env_source == "procgen":
            for env in listdir(path.join(data_root, "procgen")):
                folders.append(path.join(data_root, "procgen", env, split))
        elif env_source == "retro":
            for env in listdir(path.join(data_root, "retro")):
                folders.append(path.join(data_root, "retro", env, split))
        elif env_source == "game":
            for env in listdir(path.join(data_root, "procgen")):
                folders.append(path.join(data_root, "procgen", env, split))
            for env in listdir(path.join(data_root, "retro")):
                folders.append(path.join(data_root, "retro", env, split))
        elif env_source == "robot":
            for env in listdir(path.join(data_root, "openx")):
                folders.append(path.join(data_root, "openx", env, split))
        else:
            raise ValueError(f"Invalid source: {env_source}")
        self.subsets = []
        for folder in tqdm(folders, desc="Loading subsets..."):
            print("Subset:", folder.split("/")[-2])
            self.subsets.append(VideoDataset(split_path=folder, **kwargs))
        print("Number of subsets:", len(self.subsets))

        if sampling_strategy == "sample":
            # Sample uniformly from all samples
            probs = [len(d) for d in self.subsets]
        elif sampling_strategy == "dataset":
            # Sample uniformly from all datasets
            probs = [1 for _ in self.subsets]
        elif sampling_strategy == "log":
            # Generate probabilities according to the scale of each dataset
            probs = [math.log(len(d)) if len(d) else 0 for d in self.subsets]
        elif sampling_strategy == "pi":
            # Generate probabilities according to the scale of each dataset
            probs = [len(d) ** 0.43 for d in self.subsets]
        else:
            raise ValueError(f"Unavailable sampling strategy: {sampling_strategy}")
        total_prob = sum(probs)
        assert total_prob > 0, "No sample is available"
        self.sample_probs = [x / total_prob for x in probs]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
        index (int): Index (ignored since we sample randomly).

        Returns:
        TensorDict: Dict containing all the data blocks.
        """

        # Randomly select a subset based on weights
        subset = choices(self.subsets, self.sample_probs)[0]

        # Sample a valid sample with a random index
        sample_idx = randint(0, len(subset) - 1)
        sample_item = subset[sample_idx]
        return sample_item


class VideoDataSampler(LightningDataModule):
    def __init__(
            self,
            data_root: str,
            env_source: str = "libero",
            batch_size: int = 1,
            num_workers: int = 8,
            resolution: int = 256,
            n_context_frames: int = 5,
            prefetch_factor: int = 4,
            shuffle: bool = True,
            samples_per_epoch: int = 60000
    ):
        super(VideoDataSampler, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.shuffle = shuffle
        self.train_dataset = MultiSourceSamplerDataset(
            data_root=data_root, env_source=env_source, split="train", randomize=True,
            resolution=resolution, n_context_frames=n_context_frames, samples_per_epoch=samples_per_epoch
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )
