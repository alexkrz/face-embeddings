import os
from collections import OrderedDict
from pathlib import Path

import torch
from jsonargparse import CLI

from src.backbones.build import build_backbone


def adjust_magface_dict(model: torch.nn.Module, state_dict: OrderedDict) -> OrderedDict:
    adjusted_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.split("features.module.")[-1]
        if new_k in model.state_dict().keys() and v.size() == model.state_dict()[new_k].size():
            adjusted_dict[new_k] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(adjusted_dict.keys())
    assert num_model == num_ckpt, "Sizes of model keys and checkpoint keys do not match"
    return adjusted_dict


def main(
    method: str = "magface",
    ckpt_fp: str = "checkpoints/magface/magface_iresnet50_MS1MV2_ddp_fp32.pth",
    backbone: str = "iresnet50",
):
    model = build_backbone(backbone=backbone, embed_dim=512, pretrained=False)
    if method == "arcface":
        state_dict = torch.load(ckpt_fp)
        # print(state_dict.keys())
        model.load_state_dict(state_dict)
    elif method == "magface":
        ckpt = torch.load(ckpt_fp)
        state_dict = adjust_magface_dict(model, ckpt["state_dict"])
        # print(state_dict.keys())
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    CLI(main, as_positional=False, parser_mode="omegaconf")
