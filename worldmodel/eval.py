import cv2 as cv
import piq
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from fvd_utils.fvd_utils import get_fvd_logits, frechet_distance, load_fvd_model
from sample_utils import *

SAVE_PATH = "outputs/demo/"

ViDEO_LEN = 16


def load_video(video_path):
    cap = cv.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            # Frame was successfully read, parse it
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frames.append(frame)
        else:
            # Reach the end of video
            break
    cap.release()

    assert len(frames) == ViDEO_LEN

    video = torch.stack(frames) / 255.0
    video = rearrange(video, "t h w c -> t c h w")
    return video


if __name__ == "__main__":
    out_path = os.path.join(SAVE_PATH, "dream", "videos")
    gt_path = os.path.join(SAVE_PATH, "target", "videos")
    out_videos = os.listdir(out_path)
    gt_videos = os.listdir(gt_path)
    assert len(out_videos) == len(gt_videos)

    out_list = []
    gt_list = []
    for out_video, gt_video in tqdm(zip(out_videos, gt_videos), desc="Loading"):
        out_video_path = os.path.join(out_path, out_video)
        gt_video_path = os.path.join(gt_path, gt_video)

        out_slices = load_video(out_video_path)
        gt_slices = load_video(gt_video_path)

        out_list.append(out_slices.cpu())
        gt_list.append(gt_slices.cpu())

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
    for x_batch, y_batch in tqdm(zip(x.split(ViDEO_LEN, dim=0), y.split(ViDEO_LEN, dim=0)), desc="Evaluating"):
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

    print(f"PSNR: {sum(all_psnr) / len(all_psnr):.2f}")
    print(f"SSIM: {sum(all_ssim) / len(all_ssim):.2f}")
    print(f"LPIPS: {sum(all_lpips) / len(all_lpips):.2f}")
    print(f"FID: {sum(all_fid) / len(all_fid):.2f}")
    print(f"FVD: {fvd:.2f}")
    print(f"ES: {cos_es:.3f}")
