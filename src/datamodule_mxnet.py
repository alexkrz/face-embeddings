import numbers
import os
from pathlib import Path
from typing import Optional, Tuple

import mxnet as mx
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MXFaceDataset(Dataset):
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
        transform: transforms.Compose = default_transform,
        custom_targets: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        self.custom_targets = custom_targets
        if self.custom_targets is not None:
            assert len(self.custom_targets) == len(self.imgidx)

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, index):
        idx = self.imgidx[index]  # Looks like idx = index + 1
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        if self.custom_targets is None:
            label = header.label
            if not isinstance(label, numbers.Number):
                # In some cases header.label might contain an array and we are only interested in the first value
                label = float(label[0])
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(self.custom_targets[index], dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label


class MXFaceDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        print(f"Generating MXFaceDataset for stage {stage}")
        self.dataset = MXFaceDataset(
            root_dir=self.hparams.root_dir,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
