import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from jsonargparse import CLI
from pytorch_lightning.callbacks import ModelSummary
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.build import build_backbone
from src.datamodule_fs import IMGFaceDataset


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
    save_config: bool = False,
):
    local_vars = locals()
    # For now we need to tell Intellisense explicitly the change of variable type with type comments
    data_dir = Path(data_dir)  # type: Path
    output_dir = Path(output_dir)  # type: Path
    assert data_dir.exists()
    if not output_dir.exists():
        output_dir.mkdir()
    if data_name is None:
        data_name = data_dir.name

    if save_config:
        with open(output_dir / "predict_config.yaml", "w") as yaml_file:
            yaml.dump(local_vars, yaml_file)

    # Assign device where code is executed
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Neural Engine (MPS)
    else:
        device = torch.device("cpu")  # Default to CPU

    # Construct dataset
    print(f"Generating dataset class for {data_name}..")
    dataset = IMGFaceDataset(data_dir, file_ext, img_shape=(112, 112))
    print("Number of images in dataset:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    print(f"Loading model checkpoint for {model_name}..")
    model = build_backbone(backbone=backbone, embed_dim=512, pretrained=False)
    if "arcface" in model_name:
        state_dict = torch.load(ckpt_fp, map_location=torch.device("cpu"), weights_only=True)
        # print(state_dict.keys())
        model.load_state_dict(state_dict)
    elif "magface" in model_name:
        ckpt = torch.load(ckpt_fp, map_location=torch.device("cpu"), weights_only=True)
        state_dict = adjust_magface_dict(model, ckpt["state_dict"])
        # print(state_dict.keys())
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError()

    model.to(device)
    model.eval()

    filename_list = []
    feats_list = []
    print(f"Computing embeddings on {device}..")
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
