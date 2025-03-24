from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn

from ...modules.distributions.distributions import DiagonalGaussianDistribution


class AbstractRegularizer(nn.Module):
    def __init__(self):
        super(AbstractRegularizer, self).__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_trainable_parameters(self):
        raise NotImplementedError


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super(DiagonalGaussianRegularizer, self).__init__()
        self.sample = sample

    def get_trainable_parameters(self):
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = {}
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log
