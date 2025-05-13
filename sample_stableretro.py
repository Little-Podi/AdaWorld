import random
from dataclasses import dataclass
from os import listdir, makedirs, path

import imageio
import retro
from tqdm.auto import trange


@dataclass
class Args:
    num_logs: int = 100  # Number of episodes to generate
    timeout: int = 1000  # Timeout for generating samples
    root: str = "data"  # Root folder to save the videos


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    writer = imageio.get_writer(output_file, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()


def generate_sample(env_name: str, timeout: int, root: str, split: str, bias: int) -> None:
    env = retro.make(game=env_name, render_mode="rgb_array")

    frames = [env.reset()[0]]
    for t in range(timeout - 1):
        bias += t // 500
        action_todo = env.action_space.sample()
        if random.random() > 0.1 and bias < 4:
            action_todo[4 + bias] = 1

        obs, reward, terminated, truncated, info = env.step(action_todo)  # 60 FPS
        frames.append(obs)
        if terminated:
            frames.append(env.reset()[0])

    env.close()

    save_dir = path.join(root, "retro", env_name, split)
    makedirs(save_dir, exist_ok=True)
    current_idx = len(listdir(save_dir))
    save_path = path.join(save_dir, f"{current_idx:05}.mp4")

    save_images_to_video(frames, save_path)


env_list = retro.data.list_games()
for env_name in env_list:
    try:
        for n_log in trange(Args.num_logs,
                            desc=f"Generating {Args.num_logs} {env_name.upper()} videos for training"):
            generate_sample(env_name, Args.timeout, Args.root, "train", n_log % 5)

        for n_log in trange(Args.num_logs // 10,
                            desc=f"Generating {Args.num_logs // 10} {env_name.upper()} videos for test"):
            generate_sample(env_name, Args.timeout, Args.root, "test", n_log % 5)
    except:
        pass
