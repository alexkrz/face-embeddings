import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class IMGFaceDataset(Dataset):
    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(
        self,
        root_dir: Path,
        file_ext: str = ".jpg",
        img_shape: tuple[int, int] = (112, 112),
        transform: transforms.Compose | None = default_transform,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.file_list = sorted(list(root_dir.rglob("*" + file_ext)))
        self.img_shape = img_shape
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_p = Path(self.file_list[idx])
        relative_path = file_p.relative_to(self.root_dir)
        img = Image.open(file_p)
        # Check image shape and use resize transform if necessary
        if not img.size == self.img_shape:
            # print("Rezising image")
            # tfm = transforms.Resize(self.img_shape)
            tfm = transforms.CenterCrop(self.img_shape)
            img = tfm(img)
        # Apply additional transforms
        if self.transform:
            img = self.transform(img)
        return img, str(relative_path)
