import torch
from einops import repeat

from vwm.util import default


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, bs, num_frames, rand=None):
        rand_init = torch.randn((bs,))
        rand_init = repeat(rand_init, "b -> (b t)", t=num_frames)
        log_sigma = self.p_mean + self.p_std * default(rand, rand_init)
        return log_sigma.exp()
