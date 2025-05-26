from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from vwm.modules import UNCONDITIONAL_CONFIG
from vwm.modules.autoencoding.temporal_ae import VideoDecoder
from vwm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from vwm.modules.ema import LitEma
from vwm.util import append_dims, default, disabled_train, get_obj_from_str, instantiate_from_config


class DiffusionEngine(LightningModule):
    def __init__(
            self,
            network_config,
            denoiser_config,
            first_stage_config,
            conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            network_wrapper: Union[None, str] = None,
            use_ema: bool = False,
            ema_decay_rate: float = 0.9999,
            scale_factor: float = 1.0,
            disable_first_stage_autocast=False,
            input_key: str = "img",
            compile_model: bool = False,
            en_and_decode_n_samples_a_time: Optional[int] = None,
            n_context_frames: int = 5
    ):
        super(DiffusionEngine, self).__init__()
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config is not None else None

        self.use_ema = use_ema
        self.ema_decay_rate = ema_decay_rate
        if use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.n_context_frames = n_context_frames
        self.num_frames = n_context_frames + 1

    def reinit_ema(self):
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=self.ema_decay_rate)
            print(f"Reinitializing EMAs of {len(list(self.model_ema.buffers()))}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # Image tensors should be scaled to -1 ... 1 and in bchw format
        input_shape = batch[self.input_key].shape
        if len(input_shape) != 4:  # Is an image sequence
            assert input_shape[1] == self.num_frames
            batch[self.input_key] = rearrange(batch[self.input_key], "b t c h w -> (b t) c h w")
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = z / self.scale_factor
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for current_z in z.split(n_samples, dim=0):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": current_z.shape[0]}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(current_z, **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for current_x in x.split(n_samples, dim=0):
                out = self.first_stage_model.encode(current_x)
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = z * self.scale_factor
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)  # Go to StandardDiffusionLoss
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict):
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        x = rearrange(x, "(b t) ... -> b t ...", t=self.num_frames)
        cond_aug = append_dims(batch["context_aug"].squeeze(), x.ndim)
        mask_lens = batch["context_len"].to(dtype=torch.int)
        aug_masks = torch.arange(self.n_context_frames, device=x.device)[None].expand(len(x), -1)
        aug_masks = append_dims(aug_masks >= (self.n_context_frames - mask_lens), x.ndim)
        x[:, :self.n_context_frames] += cond_aug * torch.randn_like(x[:, :self.n_context_frames]) * aug_masks
        x = rearrange(x, "b t ... -> (b t) ...")
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx: int):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    # def on_train_epoch_start(self, *args, **kwargs) -> None:
    #     if self.conditioner.embedders[2].lam.ckpt_path and "last" in self.conditioner.embedders[2].lam.ckpt_path:
    #         self.conditioner.embedders[2].lam.reload_ckpt(self.conditioner.embedders[2].lam.ckpt_path)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", {}))

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(filter(lambda x: x.requires_grad, embedder.parameters()))
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)

        # Learning rate discount during specialized world model adaptation
        # param_dicts = [
        #     {
        #         "params": list(self.model.parameters()),
        #         "lr": lr * 0.1
        #     }
        # ]
        # for embedder in self.conditioner.embedders:
        #     if embedder.is_trainable:
        #         param_dicts.append(
        #             {
        #                 "params": list(filter(lambda x: x.requires_grad, embedder.parameters()))
        #             }
        #         )
        # opt = self.instantiate_optimizer_from_config(param_dicts, lr, self.optimizer_config)

        if self.scheduler_config is None:
            return opt
        else:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1
                }
            ]
            return [opt], scheduler

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            x_ori: torch.Tensor,
            uc: Union[Dict, None] = None,
            N: int = 10,
            shape: Union[None, Tuple, List] = None,
            **kwargs
    ):
        randn = torch.randn(N, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(self.model, input, sigma, c, **kwargs)

        samples = self.sampler(denoiser, randn, cond, x_ori=x_ori, uc=uc)
        return samples

    @torch.no_grad()
    def log_images(
            self,
            batch: Dict,
            N: int = 10,
            sample: bool = True,
            ucg_keys: List[str] = None,
            **kwargs
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders if e.ucg_rate > 0.0]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else []
        )

        sampling_kwargs = {}
        log = {}
        N = min(x.shape[0], N * self.num_frames)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        targets = self.decode_first_stage(z)
        log["targets"] = targets

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(c, x_ori=z, shape=z.shape[1:], uc=uc, N=N, **sampling_kwargs)
            samples = self.decode_first_stage(samples)
            log["samples"] = samples

            compare_list = []
            for clip_id in range(N // self.num_frames):
                compare_list.append(log["targets"][clip_id * self.num_frames:(clip_id + 1) * self.num_frames])
                compare_list.append(log["samples"][clip_id * self.num_frames:(clip_id + 1) * self.num_frames])
            log["pairs"] = torch.cat(compare_list)
        return log
