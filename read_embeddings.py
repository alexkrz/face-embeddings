import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Initialize plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = np.ravel(axs)  # Flatten to 1D array
    axs: List[plt.Axes]  # Type hinting

    # Plot Arcface distribution
    embed = np.array(np.load("results/lfw_arcface-r50.npy"))
    # print(embed.shape)
    ampl = np.linalg.norm(embed, ord=2, axis=1)
    axs[0].hist(ampl, bins=50, edgecolor="black")
    axs[0].set_title("Arcface")

    # Plot Magface distribution
    embed = np.array(np.load("results/lfw_magface-r50.npy"))
    ampl = np.linalg.norm(embed, ord=2, axis=1)
    axs[1].hist(ampl, bins=50, edgecolor="black")
    axs[1].set_title("Magface")

    plt.show()


if __name__ == "__main__":
    main()
