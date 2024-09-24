from pathlib import Path

import torch
from jsonargparse import CLI


def main(
    ckpt_p: str = "checkpoints/custom/epoch=4-step=19165.ckpt",
):
    """Convert Pytorch Lightning Checkpoint to vanilla Pytorch weights.

    Args:
        ckpt_p (str, optional): Path to .ckpt file.
    """

    checkpoint = torch.load(ckpt_p)
    print(checkpoint.keys())
    print("hyper_parameters:", checkpoint["hyper_parameters"])
    print("datamodule_hyper_parameters:", checkpoint["datamodule_hyper_parameters"])
    dataset_name = Path(checkpoint["datamodule_hyper_parameters"]["root_dir"]).name
    header_name = checkpoint["hyper_parameters"]["header"]
    backbone_name = checkpoint["hyper_parameters"]["backbone"]

    # Shorten dataset name
    if dataset_name == "casia_webface":
        dataset_name = "casia"

    backbone_weights = {
        k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("backbone.")  # fmt: skip
    }

    out_fname = f"backbone_{dataset_name}_{header_name}_{backbone_name}.pth"

    # Save weights
    torch.save(backbone_weights, Path(ckpt_p).parent / out_fname)


if __name__ == "__main__":
    CLI(main)
