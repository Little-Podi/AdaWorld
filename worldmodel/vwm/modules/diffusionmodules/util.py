"""
Partially adopted from
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
and
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
and
https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
"""

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: The function to evaluate.
    :param inputs: The argument sequence to pass to 'func'.
    :param params: A sequence of parameters 'func' depends on but does not explicitly take as arguments.
    :param flag: If False, disable gradient checkpointing.
    """

    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_params = [x.requires_grad_(True) for x in ctx.input_params]
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the Tensor
            # storage in place, which is not allowed for detach()'d Tensors
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: A 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: The dimension of the output.
    :param max_period: Controls the minimum frequency of the embeddings.
    :return: An [N x dim] Tensor of positional embeddings.
    """

    if repeat_only:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def normalization(channels):
    """
    Make a standard normalization layer.
    """

    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """

    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """

    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """

    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
            self,
            alpha: float,
            merge_strategy: str,
            rearrange_pattern: str
    ):
        super(AlphaBlender, self).__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert merge_strategy in self.strategies, f"'merge_strategy' needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            alpha = rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1")
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError
        return alpha

    def forward(
            self,
            x_spatial: torch.Tensor,
            x_temporal: torch.Tensor
    ) -> torch.Tensor:
        alpha = self.get_alpha()
        x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        return x
