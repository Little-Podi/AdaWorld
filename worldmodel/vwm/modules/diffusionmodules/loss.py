from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from vwm.modules.encoders.modules import GeneralConditioner
from vwm.util import append_dims, instantiate_from_config

from .denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    def __init__(
            self,
            sigma_sampler_config: Dict,
            loss_weighting_config: Dict,
            loss_type: str = "l2",
            n_context_frames: int = 5
    ):
        super(StandardDiffusionLoss, self).__init__()
        assert loss_type in ["l2", "l1"]
        self.loss_type = loss_type
        self.num_frames = n_context_frames + 1

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

    def get_noised_input(
            self,
            sigmas_bc: torch.Tensor,
            noise: torch.Tensor,
            input: torch.Tensor
    ) -> torch.Tensor:
        noise = rearrange(noise, "(b t) ... -> b t ...", t=self.num_frames)
        noise[:, :-1] = 0
        noise = rearrange(noise, "b t ... -> (b t) ...")
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
            self,
            network: nn.Module,
            denoiser: Denoiser,
            conditioner: GeneralConditioner,
            input: torch.Tensor,
            batch: Dict
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input)

    def _forward(
            self,
            network: nn.Module,
            denoiser: Denoiser,
            cond: Dict,
            input: torch.Tensor
    ):
        sigmas = self.sigma_sampler(input.shape[0], input.shape[0] // self.num_frames, self.num_frames).to(input)
        noise = torch.randn_like(input)
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        output = denoiser(network, noised_input, sigmas, cond)
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        output = rearrange(output, "(b t) ... -> b t ...", t=self.num_frames)[:, -1]
        input = rearrange(input, "(b t) ... -> b t ...", t=self.num_frames)[:, -1]
        w = rearrange(w, "(b t) ... -> b t ...", t=self.num_frames)[:, -1]
        return self.get_loss(output, input, w)

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
