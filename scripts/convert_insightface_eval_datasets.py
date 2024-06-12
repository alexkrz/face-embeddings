import os
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
from jsonargparse import CLI
from tqdm import tqdm


def load_bin(path: str, image_size: Tuple[int, int] = (112, 112)):
    with open(path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")

    data = np.zeros((len(issame_list) * 2, image_size[0], image_size[1], 3), dtype=np.uint8)
    print("Loading images..")
    for idx in tqdm(range(len(issame_list) * 2)):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            print("Resizing image..")
            img = mx.image.resize_short(img, image_size[0])
        data[idx] = img.asnumpy()

    return data, issame_list


def zip_directory(directory: str, zip_name: str):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in tqdm(files):
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(directory, os.pardir)),
                )


def main(
    root_dir: Path = Path(os.environ["DATASET_DIR"]) / "casia-webface-mxnet",
    out_folder: str = "zipfiles",
    img_ext: str = ".jpg",
):
    binary_files = list(root_dir.glob("*.bin"))
    binary_files = [file.name for file in binary_files]
    binary_files = sorted(binary_files, key=lambda x: x.lower())

    if not (root_dir / out_folder).exists():
        (root_dir / out_folder).mkdir()

    for file_name in binary_files:
        print(f"Processing {file_name}")

        # Set up paths
        file_p = root_dir / file_name
        out_dir = root_dir / out_folder / file_p.stem
        if not out_dir.exists():
            out_dir.mkdir()

        # Load data
        data, issame_list = load_bin(str(file_p))
        print("Number of pairs:", len(issame_list))
        num_digits = len(str(len(data)))

        # Write data
        count = 0
        refs = []
        probes = []
        mateds = []
        print("Writing images in pairs..")
        for i in tqdm(range(0, len(data), 2)):
            img1 = data[i]
            img2 = data[i + 1]
            plt.imsave(out_dir / f"{i:0{num_digits}d}{img_ext}", img1)
            plt.imsave(out_dir / f"{i+1:0{num_digits}d}{img_ext}", img2)
            refs.append(f"{i:0{num_digits}d}")
            probes.append(f"{i+1:0{num_digits}d}")
            mateds.append(issame_list[count])
            count += 1
        df_pair = pd.DataFrame(
            {
                "ref": pd.Series(refs, dtype="object"),
                "probe": pd.Series(probes, dtype="object"),
                "mated": pd.Series(mateds, dtype="int"),
            }
        )
        # print(df_pair.head())
        df_pair.to_csv(out_dir / "pair_list.txt", header=False, index=False, sep=" ")

        # Zip directory
        print("Zip directory..")
        zip_directory(str(out_dir), str(out_dir) + ".zip")

        # Remove unzipped directory
        shutil.rmtree(out_dir)


if __name__ == "__main__":
    CLI(main, as_positional=False)
