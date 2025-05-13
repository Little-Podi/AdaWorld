from os import makedirs, path
from typing import Callable, Dict, Iterable, Tuple, Union

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]

from external.lam.modules import LatentActionModel


class LAM(LightningModule):
    def __init__(
            self,
            image_channels: int = 3,
            # Latent action autoencoder
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            beta: float = 0.01,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            optimizer: OptimizerCallable = AdamW,
            ckpt_path: Union[None, str] = None
    ) -> None:
        super(LAM, self).__init__()
        self.lam = LatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout
        )
        self.beta = beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.reload_ckpt(ckpt_path)

    def reload_ckpt(self, ckpt_path: str) -> None:
        if path.exists(ckpt_path):
            lam = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = self.load_state_dict(lam, strict=False)
            print(f"Restored LAM from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing LAM keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected LAM keys: {unexpected}")
        else:
            print(f"LAM checkpoint {ckpt_path} does not exist")

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.lam(batch)
        gt_future_frames = batch["videos"][:, 1:]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        kl_loss = -0.5 * torch.sum(1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1).mean()
        loss = mse_loss + self.beta * kl_loss
        return outputs, loss, (
            ("mse_loss", mse_loss),
            ("kl_loss", kl_loss)
        )

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if batch_idx % self.log_interval == 0:  # Start of the epoch
            self.log_images(batch, outputs, "train")
        return loss

    # @torch.no_grad()
    # def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    #     # Compute the validation loss
    #     outputs, loss, aux_losses = self.shared_step(batch)
    #
    #     # Log the validation loss
    #     self.log_dict(
    #         {**{"val_loss": loss}, **{f"val/{k}": v for k, v in aux_losses}},
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=True,
    #         sync_dist=True
    #     )
    #
    #     if batch_idx % self.log_interval == 0:  # Start of the epoch
    #         self.log_images(batch, outputs, "val")
    #     return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log_images(batch, outputs, "test")
        return loss

    def log_images(self, batch: Dict, outputs: Dict, split: str) -> None:
        gt_seq = batch["videos"][0].clamp(0, 1).cpu()
        recon_seq = outputs["recon"][0].clamp(0, 1).cpu()
        recon_seq = torch.cat([gt_seq[:1], recon_seq], dim=0)
        compare_seq = torch.cat([gt_seq, recon_seq], dim=1)
        compare_seq = rearrange(compare_seq * 255, "t h w c -> h (t w) c")
        compare_seq = compare_seq.detach().numpy().astype(np.uint8)
        img_path = path.join(self.log_path, f"{split}_step{self.global_step:06}.png")
        makedirs(path.dirname(img_path), exist_ok=True)
        img = Image.fromarray(compare_seq)
        img.save(img_path)

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
