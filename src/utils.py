from pathlib import Path
from typing import Tuple, Union

import numpy as np


def find_max_version(path: Union[Path, str]):
    path = Path(path)

    version_dirs = list(path.glob("version_*"))

    if len(version_dirs) == 0:
        return -1

    version_dirs = [entry for entry in version_dirs if entry.is_dir()]

    versions = [int(str(dir_name).split("_")[-1]) for dir_name in version_dirs]
    max_version = max(versions)
    return max_version


def load_txt(fpath: Path) -> list:
    with open(fpath) as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    return lines


def create_dict(list: list, array: np.ndarray) -> dict:
    result_dict = {}
    for item, row in zip(list, array):
        result_dict[item] = row
    return result_dict


def process_names(names: list) -> Tuple[np.ndarray, np.ndarray]:
    names = np.array(names)
    # Filename has structure parent_dir / person_id / sample_id .{file_ext}
    # Split each filename at the "/" string and expand dimensions
    split_arr = np.stack(np.char.split(names, sep="/"))
    # Get person_ids from index -2 and convert to int
    person_ids = split_arr[:, -2]
    person_ids = np.array(person_ids, dtype=int)
    # Get sample_ids from index -1, remove .{file_ext} ending and convert to int
    sample_ids = split_arr[:, -1]
    sample_ids = np.stack(np.char.split(sample_ids, sep="."))
    sample_ids = sample_ids[:, 0]
    sample_ids = np.array(sample_ids, dtype=int)
    return person_ids, sample_ids
