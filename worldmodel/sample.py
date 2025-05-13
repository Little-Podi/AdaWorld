import cv2 as cv
import piq
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch import autocast
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from fvd_utils.fvd_utils import get_fvd_logits, frechet_distance, load_fvd_model
from sample_utils import *

SAVE_PATH = "outputs/demo/"
CONFIG = "configs/inference/adaworld.yaml"
CKPT = "path_to/adaworld.safetensors"

RESOLUTION = 256
AUG_LEVEL = 0.1
NUM_STEPS = 5
CFG_SCALE = 1.05
ViDEO_LEN = 20
CONTEXT_FRAME = 6

source_list = []
procgen_envs = os.listdir("../data/procgen")
for procgen_env in procgen_envs:
    source_list.append({
        "file_name": f"../data/procgen/{procgen_env}/test/00000.mp4",
        "start_ind": 50
    })
retro_envs = os.listdir("../data/retro")
for retro_env in retro_envs:
    source_list.append({
        "file_name": f"../data/retro/{retro_env}/test/00000.mp4",
        "start_ind": 50
    })
target_list = source_list


def load_video_slices(video_path, start_id: int = 0, frame_skip: int = 1):
    cap = cv.VideoCapture(video_path)
    if "retro" in video_path:
        frame_skip = 4
    elif "procgen" not in video_path and "ssv2" not in video_path and "mira" not in video_path:
        frame_skip = 2
    num_frames = ViDEO_LEN * frame_skip

    cap.set(cv.CAP_PROP_POS_FRAMES, start_id)
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
            frames.extend([frames[-1]] * (num_frames - len(frames)))
            break
    cap.release()

    video = torch.stack(frames[::frame_skip]) / 255.0

    # Crop the video to be square
    if video.shape[1] != video.shape[2]:
        square_len = min(video.shape[1], video.shape[2])
        h_crop = (video.shape[1] - square_len) // 2
        w_crop = (video.shape[2] - square_len) // 2
        video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

    if video.shape[-2] != RESOLUTION or video.shape[-3] != RESOLUTION:
        video = rearrange(video, "t h w c -> t c h w")
        video = F.interpolate(video, RESOLUTION, mode="bicubic")
    else:
        video = rearrange(video, "t h w c -> t c h w")
    return [frame for frame in video]


def get_sample(source_video_dict, target_video_dict):
    source_image_seq = load_video_slices(source_video_dict["file_name"], source_video_dict["start_ind"])
    target_image_seq = load_video_slices(target_video_dict["file_name"], target_video_dict["start_ind"])
    lam_inputs = torch.stack(source_image_seq[:2] + [target_image_seq[0]])
    filled_seq = [torch.zeros_like(target_image_seq[0])] * (CONTEXT_FRAME - 1) + target_image_seq
    gt_frames = torch.stack(filled_seq[CONTEXT_FRAME - 1:])
    next_frames = torch.Tensor(filled_seq[CONTEXT_FRAME])
    prev_frames = torch.stack(filled_seq[:CONTEXT_FRAME])
    next_frames = next_frames * 2.0 - 1.0
    prev_frames = prev_frames * 2.0 - 1.0
    img_seq = torch.cat([prev_frames, next_frames[None]])

    context_len = torch.Tensor([1])
    context_aug = torch.Tensor([AUG_LEVEL])

    value_dict = {
        "source_video": torch.stack(source_image_seq).to("cuda"),
        "gt_frames": gt_frames.to("cuda"),
        "img_seq": img_seq.to("cuda"),
        "cond_frames_without_noise": prev_frames[-1][None].to("cuda"),
        "cond_frames": (prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]))[None].to("cuda"),
        "lam_inputs": lam_inputs[None].to("cuda"),
        "context_len": context_len.to("cuda"),
        "context_aug": context_aug.to("cuda")
    }
    return value_dict


def run_fdm(fdm_model, source_video_dict, target_video_dict):
    value_dict = get_sample(source_video_dict, target_video_dict)
    sampler = init_sampling(steps=NUM_STEPS, cfg_scale=CFG_SCALE, n_context_frames=CONTEXT_FRAME)

    out = do_sample(
        fdm_model,
        sampler,
        value_dict,
        input_res=RESOLUTION,
        force_uc_zero_embeddings=["cond_frames_without_noise", "cond_frames", "lam_inputs"]
    )
    return out, value_dict["gt_frames"], value_dict["source_video"]


if __name__ == "__main__":
    seed_everything(32)

    model = init_model(CONFIG, CKPT)

    out_list = []
    gt_list = []
    sampling_process = tqdm(total=len(source_list), desc="Dreaming")

    with torch.no_grad(), autocast("cuda"), model.ema_scope("Sampling"):
        for source_item, target_item in zip(source_list, target_list):
            samples, gt_images, source_video = run_fdm(model, source_item, target_item)

            out_list.append(samples.cpu())
            gt_list.append(gt_images[1:].cpu())

            samples = torch.cat([gt_images[:1], samples], dim=0)
            samples = samples.clamp(0, 1)
            gt_images = gt_images.clamp(0, 1)
            source_video = source_video.clamp(0, 1)
            perform_save_locally(os.path.join(SAVE_PATH, "dream"), samples, "grids")
            perform_save_locally(os.path.join(SAVE_PATH, "target"), gt_images, "grids")
            perform_save_locally(os.path.join(SAVE_PATH, "source"), source_video, "grids")
            perform_save_locally(os.path.join(SAVE_PATH, "dream"), samples, "videos")
            perform_save_locally(os.path.join(SAVE_PATH, "target"), gt_images, "videos")
            perform_save_locally(os.path.join(SAVE_PATH, "source"), source_video, "videos")

            sampling_process.update(1)

    model.cpu()
    torch.cuda.empty_cache()

    outs = torch.cat(out_list, dim=0)
    gts = torch.cat(gt_list, dim=0)

    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_fid = []
    fvd_real = []
    fvd_fake = []
    x = gts.clamp(0, 1)
    y = outs.clamp(0, 1)
    fid_model = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=True)
    i3d_model = load_fvd_model(torch.device("cuda"))
    for x_batch, y_batch in zip(x.split(ViDEO_LEN - 1, dim=0), y.split(ViDEO_LEN - 1, dim=0)):
        psnr = piq.psnr(x_batch, y_batch).mean()
        ssim = piq.ssim(x_batch, y_batch).mean()
        lpips = piq.LPIPS()(x_batch, y_batch).mean()

        x_batch_299 = F.interpolate(x_batch, 299, mode="bicubic")
        y_batch_299 = F.interpolate(y_batch, 299, mode="bicubic")
        fid_model.update(x_batch_299, real=True)
        fid_model.update(y_batch_299, real=False)
        fid = fid_model.compute()
        fid_model.reset()

        x_batch_i3d = (x_batch.permute(0, 2, 3, 1) * 255.0)[None].int().data.numpy()
        y_batch_i3d = (y_batch.permute(0, 2, 3, 1) * 255.0)[None].int().data.numpy()
        fvd_fake.append(get_fvd_logits(x_batch_i3d, i3d_model, torch.device("cuda"), batch_size=1))
        fvd_real.append(get_fvd_logits(y_batch_i3d, i3d_model, torch.device("cuda"), batch_size=1))

        all_psnr.append(psnr.item())
        all_ssim.append(ssim.item())
        all_lpips.append(lpips.item())
        all_fid.append(fid.item())

    fvd_fake = torch.cat(fvd_fake)
    fvd_real = torch.cat(fvd_real)
    fvd, cos_es = frechet_distance(fvd_fake, fvd_real)

    print(f"PSNR: {sum(all_psnr) / len(all_psnr):.3f}")
    print(f"SSIM: {sum(all_ssim) / len(all_ssim):.3f}")
    print(f"LPIPS: {sum(all_lpips) / len(all_lpips):.3f}")
    print(f"FID: {sum(all_fid) / len(all_fid):.3f}")
    print(f"FVD: {fvd:.3f}")
    print(f"ES: {cos_es:.3f}")
