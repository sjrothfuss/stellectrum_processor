"""A script to process, plot, and save fluorescence stellectra data.

SJR 202406"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

##################### OPTIONS #####################
### Input and output directories
ROOT_DIR = f"{os.getcwd()}/RawData"  # Root directory of experimental data directories
OUTPUT_DIR = f"{os.getcwd()}/Output"  # Output directory for processed data

### Image processing options
PIXELS_TO_SUBTRACT = [  # Coordinates of hot pixels to subtract from all images, if any
    (65, 69),
]
LAMBDA_2_RANGE = (485, 581)  # Start and end of lambda2 (usually excitation) in nm
LAMBDA_2_STEP = 2  # Stepsize of lambda2 in nm
# Since filenames do not include lambda2 wavelengths, they are calculated from range and stepsize

### Output options
WILL_SAVE_SPECTRA = False  # Save spectra as one .csv file per experiment
WILL_PLOT_SPECTRA = True  # Plot spectra for max excitation and emission

###################################################

# Parse inputs
ROOT_DIR = Path(ROOT_DIR)
if not ROOT_DIR.exists():
    raise FileNotFoundError(f"Root directory {ROOT_DIR} not found.")
OUTPUT_DIR = Path(OUTPUT_DIR)
if LAMBDA_2_RANGE[0] >= LAMBDA_2_RANGE[1]:
    msg = "Invalid LAMBDA_2_RANGE, second element must be greater than first."
    raise ValueError(msg)
if LAMBDA_2_STEP <= 0:
    msg = "Invalid LAMBDA_2_STEP, must be greater than 0."
    raise ValueError(msg)
if not isinstance(PIXELS_TO_SUBTRACT, list) or not all(
    isinstance(p, tuple) for p in PIXELS_TO_SUBTRACT
):
    raise TypeError("Pixels to subtract must be a list of tuples.")


def stellectrum_generator():
    """Top level function to parse, save, and plot data."""
    for root_dir, _, files in os.walk(ROOT_DIR):
        files = [f for f in files if "LA" in f and f.endswith(".tif")]
        if files:
            root_dir = Path(root_dir)
            experiment_name = root_dir.stem
            print(f"Found {len(files)} files to process in {experiment_name}.")
            dir_intensities = _parse_experimental_dir(root_dir, files)
            df = _sort_and_save_data_frame(experiment_name, dir_intensities)
            _plot_max_spectra(experiment_name, df)
            print(f"Finished processing {experiment_name}!")
    print("All experiments processed!")


def _parse_experimental_dir(
    root_dir: Path,
    files: list[str],
) -> dict[int, dict[int, int]]:
    """Parse all .tif files in a directory and return a dictionary of intensities."""
    fluorescence_data = {}
    n, c = len(files), 0
    for file_name in files:
        lambda1, lambda2 = _parse_lambda1_lambda2(file_name)
        file_path = Path(f"{root_dir}/{file_name}")
        intensity = _process_tif_file(file_path)

        if lambda1 not in fluorescence_data:
            fluorescence_data[lambda1] = {}
        fluorescence_data[lambda1][lambda2] = intensity

        c = _progress_counter(n, c)

    return fluorescence_data


def _progress_counter(n: int, c: int):
    """Print progress of the processing."""
    if c > 0 and c % 100 == 0:
        print(f"Processed {c} of {n} files...")
    return c + 1


def _parse_lambda1_lambda2(file_name: str) -> tuple[int]:
    """Parse lambda1 and lambda2 from a .tif file name and labmda2 range."""
    file_name = file_name.split("_")

    lambda1 = int(file_name[1][:-2])

    lambda2_idx = int(file_name[2][2:])
    lambda2_range = np.arange(
        LAMBDA_2_RANGE[0], LAMBDA_2_RANGE[1] + 1, LAMBDA_2_STEP, dtype=int
    )
    lambda2 = lambda2_range[lambda2_idx]

    return lambda1, lambda2


def _process_tif_file(file_path: Path) -> int:
    """Process a .tif file and return the sum of pixel intensities."""
    # print(file_path)
    tif = np.array(Image.open(file_path))
    tif = _normalize_standardize_tif(tif)
    return np.sum(tif)


def _normalize_standardize_tif(tif: np.ndarray) -> np.ndarray:
    """Placeholder function for any image modification steps."""
    tif = _subtract_hot_pixels(tif)
    # Add other standardization steps here if desired
    return tif


def _subtract_hot_pixels(tif: np.ndarray) -> np.ndarray:
    """Replace hot pixels with the median of the image."""
    for pixel in PIXELS_TO_SUBTRACT:
        x, y = pixel
        if 0 < x < tif.shape[0] - 1 and 0 < y < tif.shape[1] - 1:
            tif[x - 1 : x + 2, y - 1 : y + 2] = np.median(tif)
        else:
            print(f"Hot pixel index {pixel} out of bounds, skipping.")
    return tif


def _sort_and_save_data_frame(experiment_name: str, intensities: dict) -> pd.DataFrame:
    """Create and sort df and optionally save it as a .csv file."""
    df = pd.DataFrame.from_dict(intensities, orient="index")
    df.index.name = "Lambda1"
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)

    if not WILL_SAVE_SPECTRA:
        return df
    if OUTPUT_DIR.exists() is False:
        os.mkdir(OUTPUT_DIR)
    print("Saving intensities as {experiment_name}.csv...")
    try:
        df.to_csv(f"{OUTPUT_DIR}/{experiment_name}.csv")
    except Exception as e:
        print(f"Error saving {experiment_name}.csv!!!\n{e}")
    return df


def _plot_max_spectra(experiment_name: str, df: pd.DataFrame):
    """Plot the exictation spectra for the max emission and vice versa."""
    if not WILL_PLOT_SPECTRA:
        return
    max_loc = df.stack().idxmax()
    max_lambda1 = df.loc[max_loc[0]]
    max_lambda2 = df[max_loc[1]]
    plt.plot(max_lambda1)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"{experiment_name}: Lambda 1 spectrum at {max_loc[0]} nm")
    plt.show()

    plt.plot(max_lambda2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"{experiment_name}: Lambda 2 spectrum at {max_loc[1]} nm")
    plt.show()


stellectrum_generator()
