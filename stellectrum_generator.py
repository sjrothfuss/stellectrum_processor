import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


ROOT_DIR = f"{os.getcwd()}/RawData"
PIXELS_TO_SUBTRACT = []


def stellectrum_generator():
    fluorescence_data = {}
    for root_dir, _, files in os.walk(ROOT_DIR):
        files = [f for f in files if f.endswith(".tif")]
        if files:
            _parse_experimental_dir(root_dir, files, fluorescence_data)
    # Sort data structure and save/plot


def _parse_experimental_dir(root_path, files, fluorescence_data):
    experiment_name = root_path.split("/")[-1]
    # Initialize data structure
    for f in files:
        file_path = f"{root_path}/{f}"
        _process_tif_file(experiment_name, file_path)
        # Add to data structure
    # Return data


def _process_tif_file(experiment_name, file_path):
    tif = np.array(Image.open(file_path))
    plt.imshow(tif, cmap="gray")
    plt.show()
    tif = _subtract_hot_pixels(tif)
    # Sum pixel values
    # Store somehow
    pass


def _subtract_hot_pixels(tif):
    # Clean up and subtract hot pixels
    if not PIXELS_TO_SUBTRACT:
        return tif
    else:
        for pixel in PIXELS_TO_SUBTRACT:
            tif[pixel] = 0
        return tif


stellectrum_generator()
