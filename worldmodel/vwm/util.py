import functools
import importlib
from inspect import isfunction

import torch
from einops import repeat


def repeat_img_seq(x, num_frames):
    return repeat(x, "b ... -> (b t) ...", t=num_frames)


def disabled_train(self):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore.
    """

    return self


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
                enabled=enabled,
                dtype=torch.get_autocast_gpu_dtype(),
                cache_enabled=torch.is_autocast_cache_enabled()
        ):
            return f(*args, **kwargs)

    return do_autocast


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if val is None:
        return d() if isfunction(d) else d
    else:
        return val


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        else:
            raise KeyError("Expected key 'target' to instantiate")
    else:
        return get_obj_from_str(config["target"])(**config.get("params", {}))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """
    Appends dimensions to the end of a tensor until it has target_dims dimensions.
    """

    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"Input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]
