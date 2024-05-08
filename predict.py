import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from jsonargparse import CLI
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.build import build_backbone
from src.datamodule import IMGFaceDataset


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
    data_dir: str,
    output_dir: str,
    file_ext: str = ".jpg",
    data_name: Optional[str] = None,
    model_name: str = "arcface",
    ckpt_fp: str = "checkpoints/arcface/backbone_ms1mv3_arcface_r50_fp16.pth",
    backbone: str = "iresnet50",
    gpu: Optional[int] = None,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    assert data_dir.exists()
    assert output_dir.exists()
    if data_name is None:
        data_name = data_dir.name

    # Assign device where code is executed
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # Only show pytorch the selected GPU
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")  # Use CPU

    # Construct dataset
    print(f"Generating dataset class for {data_name}..")
    dataset = IMGFaceDataset(data_dir, file_ext, img_shape=(112, 112))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    print(f"Loading model checkpoint for {model_name}..")
    model = build_backbone(backbone=backbone, embed_dim=512, pretrained=False)
    if model_name == "arcface":
        state_dict = torch.load(ckpt_fp)
        # print(state_dict.keys())
        model.load_state_dict(state_dict)
    elif model_name == "magface":
        ckpt = torch.load(ckpt_fp)
        state_dict = adjust_magface_dict(model, ckpt["state_dict"])
        # print(state_dict.keys())
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError()

    model.to(device)
    model.eval()

    filename_list = []
    feats_list = []
    print("Computing embeddings..")
    for batch_data in tqdm(dataloader):
        imgs, filenames = batch_data
        with torch.no_grad():
            feats = model(imgs.to(device))
        feats = feats.cpu().numpy()  # Move feats to CPU and convert to numpy array
        filename_list += filenames
        feats_list.append(feats)

    print("Saving results..")
    feats_arr = np.concatenate(feats_list, axis=0)
    print("Shape of feats_arr:", feats_arr.shape)
    print("Dtype of feats_arr:", feats_arr.dtype)
    # Save feats as .npy file
    np.save(output_dir / f"{data_name}_{model_name}.npy", feats_arr)
    # Save filenames as .txt file
    with open(output_dir / f"{data_name}.txt", "w") as outfile:
        outfile.write("\n".join(filename_list))
        outfile.write("\n")  # Add final newline


if __name__ == "__main__":
    CLI(main, as_positional=False, parser_mode="omegaconf")
