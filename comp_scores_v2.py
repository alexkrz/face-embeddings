import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from jsonargparse import CLI
from tqdm import tqdm

sys.path.append(os.path.abspath("../"))
from src.utils import load_txt


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    distance = np.linalg.norm(vec1 - vec2)
    return distance


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def compute_comparison_scores(
    df: pd.DataFrame,
    feats: np.ndarray,
    filenames: list,
    score_func: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
) -> np.ndarray:
    scores = []
    for row in tqdm(range(len(df))):
        ref = df.iloc[row]["ref"]
        probe = df.iloc[row]["probe"]
        if ref in filenames:
            ref_idx = filenames.index(ref)
            ref_vec = feats[ref_idx]
        else:
            ref_vec = None
        if probe in filenames:
            probe_idx = filenames.index(probe)
            probe_vec = feats[probe_idx]
        else:
            probe_vec = None

        if ref_vec is not None and probe_vec is not None:
            score = score_func(ref_vec, probe_vec)
        else:
            score = np.nan
        scores.append(score)
    return np.array(scores)


def main(
    data_dir: Path = Path(os.environ["DATASET_DIR"]) / "EURECOM_Kinect_Face_Dataset_aligned",
    embed_dir: Path = Path(os.environ["DATASET_DIR"]) / "Face-Embeddings" / "eurecom_arcface",
    model: str = "arcface-r100",
    dataset: str = "eurecom",
):
    # out_dir = embed_dir / "comparison_scores"
    # if not out_dir.exists():
    #     out_dir.mkdir()

    df = pd.read_csv(
        data_dir / "comparisonList.txt",
        sep=";",
        names=["ref", "probe"],
        dtype={"ref": str, "probe": str},
    )
    df = df.sort_values(by="ref", ignore_index=True)
    print(df.head())
    print(len(df))

    feats = np.load(embed_dir / f"{dataset}_{model}.npy")
    filenames = load_txt(embed_dir / f"{dataset}.txt")
    # filenames = np.array(filenames)
    # Split by / in case filenames contain subdirectories
    # split_arr = np.stack(np.char.split(filenames, sep="/"))
    # filenames = split_arr[:, -1]
    # # Remove the filename extension from filenames list
    # split_arr = np.stack(np.char.split(filenames, sep="."))
    # filenames = split_arr[:, 0].tolist()

    print(f"Processing {dataset}..")
    scores = compute_comparison_scores(
        df=df,
        feats=feats,
        filenames=filenames,
        score_func=cosine_similarity,
    )
    df["mated"] = True
    df["score"] = scores

    df.to_csv(
        embed_dir / "comparison_scores.csv", float_format="%.6f", header=True, index=False, sep=";"
    )


if __name__ == "__main__":
    CLI(main, as_positional=False, parser_mode="omegaconf")
