import io
import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Function to recursively list all groups and items in an HDF5 file
def visit_file(hdf5_file: h5py.File, return_first: bool = False):
    groups = []
    data = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            data.append(name)
            if return_first:
                return True  # Stop visiting file
        elif isinstance(obj, h5py.Group):
            groups.append(name)

    hdf5_file.visititems(visitor)
    return groups, data


class H5FaceDataset(Dataset):
    default_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(
        self,
        root_dir: str,
        filename: str,
        transform: transforms.Compose = default_transform,
        custom_targets: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        hdf5_fp = os.path.join(root_dir, filename)
        self.h5file = h5py.File(hdf5_fp, "r")
        # NOTE: One could also use a separate index.txt file to avoid parsing the whole hdf5 file once
        groups, data = visit_file(self.h5file)
        self.data = data

        self.custom_targets = custom_targets
        if self.custom_targets is not None:
            assert len(self.custom_targets) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hf_data = np.array(self.h5file[self.data[idx]])
        label = int(self.data[idx].split("/")[0])
        label = torch.tensor(label, dtype=torch.long)
        image = Image.open(io.BytesIO(hf_data))

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class H5FaceDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        filename: str,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        print("Generating H5FaceDataset..")
        self.dataset = H5FaceDataset(
            root_dir=self.hparams.root_dir,
            filename=self.hparams.filename,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
