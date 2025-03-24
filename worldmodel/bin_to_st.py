import os

import torch
from safetensors.torch import save_file

ckpt = "path_to/pytorch_model.bin"

model_bin = torch.load(ckpt, map_location="cpu")  # Only contains model weights

for k in list(model_bin.keys()):  # Remove the prefix
    if "_forward_module" in k and "decay" not in k and "num_updates" not in k:
        model_bin[k.replace("_forward_module.", "")] = model_bin[k]
    del model_bin[k]

for k in list(model_bin.keys()):  # Combine EMA weights
    if "model_ema" in k:
        orig_k = None
        for kk in list(model_bin.keys()):
            if "model_ema" not in kk and k[10:] == kk[6:].replace(".", ""):
                orig_k = kk
        assert orig_k is not None
        model_bin[orig_k] = model_bin[k]
        del model_bin[k]
        print("Replace", orig_k, "with", k)

model_st = {}
for k in list(model_bin.keys()):
    model_st[k] = model_bin[k]

os.makedirs("ckpts", exist_ok=True)
save_file(model_st, "ckpts/adaworld.safetensors")
