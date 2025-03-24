from dataclasses import dataclass
from os import makedirs, path

import imageio
from gym3 import types_np
from procgen import ProcgenGym3Env
from tqdm import tqdm


@dataclass
class Args:
    num_logs: int = 10000  # Number of episodes to generate
    timeout: int = 1000  # Timeout for generating samples
    root: str = "data"  # Root folder to save the videos


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    writer = imageio.get_writer(output_file, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()


def generate_sample(env_name: str, start_level: int, timeout: int, root: str, split: str) -> None:
    env = ProcgenGym3Env(
        env_name=env_name,
        num=1,
        num_levels=1,
        start_level=start_level,
        use_sequential_levels=False,
        distribution_mode="hard",
        render_mode="rgb_array"
    )

    frames = [env.get_info()[0]["rgb"]]
    for _ in range(timeout - 1):
        action_todo = types_np.sample(env.ac_space, bshape=(env.num,))
        env.act(action_todo)  # 15 FPS
        frames.append(env.get_info()[0]["rgb"])

    env.close()

    save_path = path.join(root, "procgen", env_name, split, f"{start_level:05}.mp4")
    makedirs(path.dirname(save_path), exist_ok=True)

    save_images_to_video(frames, save_path)


env_list = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot"
]
for env_name in env_list:
    for i in tqdm(range(Args.num_logs // 10, Args.num_logs),
                  desc=f"Generating {Args.num_logs // 10 * 9} {env_name.upper()} videos for training"):
        generate_sample(env_name, i, Args.timeout, Args.root, "train")

    for i in tqdm(range(Args.num_logs // 10),
                  desc=f"Generating {Args.num_logs // 10} {env_name.upper()} videos for test"):
        generate_sample(env_name, i, Args.timeout, Args.root, "test")
