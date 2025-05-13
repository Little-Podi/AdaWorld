import os
from dataclasses import dataclass
from os import listdir, path, makedirs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import imageio
import tensorflow_datasets as tfds
from tqdm.auto import trange


@dataclass
class Args:
    save_root: str = "path_to/openx"  # Root for the videos to save
    orig_root: str = "path_to/rtx"  # Root for the downloaded videos


def dataset2path(dataset_name) -> str:
    versions = listdir(path.join(Args.orig_root, dataset_name))
    versions.sort()
    versions = [version for version in versions if len(version) == 5]
    version = versions[-1]
    return path.join(Args.orig_root, dataset_name, version)


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    writer = imageio.get_writer(output_file, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()


def extract_sample(tfds_builder, obs_key: str, dataset_name: str, save_dir: str, split: str, extra: str = None) -> None:
    try:
        ds = tfds_builder.as_dataset(split=split)
    except:
        ds = tfds_builder.as_data_source(split=split)
    ds_iter = iter(ds)
    for episode_idx in trange(len(ds), desc=f"Extracting {dataset_name.upper()} {split}"):
        try:
            episode = next(ds_iter)
            if dataset_name == "robot_vqa":
                images = []
                for step in episode["steps"]:
                    images.extend([img for img in step["observation"][obs_key]])
            elif extra is None:
                images = [step["observation"][obs_key] for step in episode["steps"]]
            else:
                images = [step["observation"][obs_key][extra] for step in episode["steps"]]
            try:
                images = [image.numpy() for image in images]
            except:
                images = images

            save_path = path.join(save_dir, split, f"{episode_idx:08}.mp4")
            makedirs(path.dirname(save_path), exist_ok=True)

            save_images_to_video(images, save_path)
        except:
            pass


dataset_list = [
    "aloha_mobile",
    "asu_table_top_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "berkeley_autolab_ur5",
    "berkeley_cable_routing",
    "berkeley_fanuc_manipulation",
    "berkeley_gnm_cory_hall",
    "berkeley_gnm_recon",
    "berkeley_gnm_sac_son",
    "berkeley_mvp_converted_externally_to_rlds",
    "berkeley_rpt_converted_externally_to_rlds",
    "bridge",
    "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "cmu_play_fusion",
    "cmu_playing_with_food",
    "cmu_stretch",
    "columbia_cairlab_pusht_real",
    "conq_hose_manipulation",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
    "dlr_sara_pour_converted_externally_to_rlds",
    "dobbe",
    "droid",
    "fmb",
    "fractal20220817_data",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "imperialcollege_sawyer_wrist_cam",
    "io_ai_tech",
    "jaco_play",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "kuka",
    "language_table",
    "maniskill_dataset_converted_externally_to_rlds",
    "mimic_play",
    "nyu_door_opening_surprising_effectiveness",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "nyu_rot_dataset_converted_externally_to_rlds",
    "plex_robosuite",
    "qut_dexterous_manpulation",
    "robo_net",
    "robo_set",
    "robot_vqa",
    "roboturk",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "stanford_mask_vit_converted_externally_to_rlds",
    "taco_play",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "toto",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "uiuc_d3field",
    "utaustin_mutex",
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "viola"
]
feasible_datasets = 0
infeasible_datasets = []
display_keys = [
    "image", "wrist_image", "hand_image", "top_image", "wrist225_image", "wrist45_image", "image_manipulation",
    "highres_image", "finger_vision_1", "finger_vision_2", "image_fisheye", "wrist_image_left",
    "image_side_1", "image_side_2", "image_wrist_1", "image_wrist_2", "image_additional_view",
    "image_left_side", "image_right_side", "image_left", "image_right", "image_top", "image_wrist",
    "front_image_1", "front_image_2", "exterior_image_1_left", "exterior_image_2_left",
    "frontleft_fisheye_image", "frontright_fisheye_image", "hand_color_image",
    "rgb", "front_rgb", "agentview_rgb", "eye_in_hand_rgb", "rgb_static", "rgb_gripper",
    "image_1", "image_2", "image_3", "image_4", "image1", "image2", "images",
    "cam_high", "cam_left_wrist", "cam_right_wrist"
]
for dataset in dataset_list:
    is_feasible = False
    builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
    for display_key in display_keys:
        if display_key in builder.info.features["steps"]["observation"]:
            if dataset == "mimic_play":
                if display_key == "image":
                    folder = path.join(Args.save_root, f"{dataset}-front_image_1")
                    if not path.exists(folder):
                        for split_name in builder.info.splits.keys():
                            extract_sample(builder, display_key, dataset, folder, split_name, "front_image_1")
                    folder = path.join(Args.save_root, f"{dataset}-front_image_2")
                    if not path.exists(folder):
                        for split_name in builder.info.splits.keys():
                            extract_sample(builder, display_key, dataset, folder, split_name, "front_image_2")
                else:
                    folder = path.join(Args.save_root, f"{dataset}-{display_key}")
                    if not path.exists(folder):
                        for split_name in builder.info.splits.keys():
                            extract_sample(builder, display_key, dataset, folder, split_name, display_key)
            else:
                folder = path.join(Args.save_root, f"{dataset}-{display_key}")
                if not path.exists(folder):
                    for split_name in builder.info.splits.keys():
                        extract_sample(builder, display_key, dataset, folder, split_name)
            is_feasible = True
    if is_feasible:
        feasible_datasets += 1
    else:
        infeasible_datasets.append(dataset)
print("Feasible datasets:", feasible_datasets)
print("Infeasible datasets:", infeasible_datasets)
