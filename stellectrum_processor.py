"""A script to process, plot, and save fluorescence stellectra data.

This script is designed to take a directory containing subdirectories of
fluorescence data recorded as .tifs. The script will process each .tif
file, sum the pixel intensities, and save the data as a .csv file. The
script will also plot the excitation and emission spectra for the
maximum intensity of each experiment.

SJR 202406"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

##################### OPTIONS #####################
### Input and output directories
ROOT_DIR = "/Users/spencer/Library/CloudStorage/OneDrive-Vanderbilt/Documents - wan_lab_vanderbilt/Wan Lab - Spencer/Projects SJR/CryoSpectra/RawData/"  # Root directory of experimental data directories
OUTPUT_DIR = f"{ROOT_DIR}/../Output/"  # Output directory for processed data

### Image processing options
PIXELS_TO_SUBTRACT = [  # Coordinates of hot pixels to subtract from all images, if any
    (65, 69),
]
# Since filenames do not include lambda2 wavelengths, they are calculated from range and stepsize
LAMBDA_2_RANGE = {  # Start and end of lambda2 (usually excitation) in nm, use a dict with format {directory_name: (start, end)}
    "20240530-cryo-GFP": (485, 581),
}
LAMBDA_2_STEP = 2  # Stepsize of lambda2 in nm, if stepsize varies among experiments, use a dict with format {directory_name: stepsize}

### Output options
WILL_SAVE_DATA = True  # Save fluorescence intensties as one .csv file per experiment
WILL_PLOT_SPECTRA = True  # Plot spectra for max excitation and emission
PLOT_FORMAT = "individual"  # "subplots" to show both ex/em on one figure or "individual" to show separately

###################################################

# Check inputs
## Check dirs
ROOT_DIR = Path(ROOT_DIR).resolve()
if not ROOT_DIR.exists():
    raise FileNotFoundError(f"Root directory {ROOT_DIR} not found.")
OUTPUT_DIR = Path(OUTPUT_DIR).resolve()
## Check LAMBDA_2_STEP
if isinstance(LAMBDA_2_STEP, (int, float)):
    if LAMBDA_2_STEP <= 0:
        msg = "Invalid LAMBDA_2_STEP, must be greater than 0."
        raise ValueError(msg)
elif isinstance(LAMBDA_2_STEP, dict):
    if not all(isinstance(s, (int, float)) for s in LAMBDA_2_STEP.values()):
        msg = "All LAMBDA_2_STEP values must be numbers."
        raise TypeError(msg)
else:
    msg = (
        f"LAMBDA_2_STEP is {type(LAMBDA_2_STEP).__name__}, must be a number or a dict."
    )
    raise TypeError(msg)
## Check LAMBDA_2_RANGE
if isinstance(LAMBDA_2_RANGE, (tuple, list)):
    if len(LAMBDA_2_RANGE) != 2:
        msg = "LAMBDA_2_RANGE lists or tuples must be of length 2."
        raise ValueError(msg)
    if LAMBDA_2_RANGE[0] >= LAMBDA_2_RANGE[1]:
        msg = "Invalid LAMBDA_2_RANGE, second element must be greater than first."
        raise ValueError(msg)
elif isinstance(LAMBDA_2_RANGE, dict):
    if not all(
        isinstance(r, (list, tuple)) and len(r) == 2 and r[0] < r[1]
        for r in LAMBDA_2_RANGE.values()
    ):
        msg = "LAMBDA_2_RANGE dict values must be lists or tuples of length 2 with first element less than second."
        raise TypeError(msg)
    for r in LAMBDA_2_RANGE.values():
        for v in r:
            if not isinstance(v, (int, float)):
                msg = (
                    f"All LAMBDA_2_RANGE values must be numbers not {type(v).__name__}."
                )
                raise TypeError(msg)
else:
    msg = f"LAMBDA_2_RANGE type is {type(LAMBDA_2_RANGE).__name__}, must be a list, tuple, or dict."
    raise TypeError(msg)
## Check PIXELS_TO_SUBTRACT
if not isinstance(PIXELS_TO_SUBTRACT, list) or not all(
    isinstance(p, tuple) for p in PIXELS_TO_SUBTRACT
):
    raise TypeError("Pixels to subtract must be a list of tuples.")
## Check output options
if PLOT_FORMAT not in ["subplots", "individual"]:
    raise ValueError("PLOT_FORMAT must be 'subplots' or 'individual'.")
## Assert types
assert isinstance(ROOT_DIR, Path)
assert isinstance(OUTPUT_DIR, Path)
assert isinstance(LAMBDA_2_RANGE, (dict, tuple, list))
assert isinstance(LAMBDA_2_STEP, (dict, int, float))
assert isinstance(PIXELS_TO_SUBTRACT, list) and all(
    isinstance(p, tuple) for p in PIXELS_TO_SUBTRACT
)


def stellectrum():
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
    experiment_name = root_dir.stem

    for file_name in files:
        lambda1, lambda2 = _parse_lambda1_lambda2(file_name, experiment_name)
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


def _parse_lambda1_lambda2(file_name: str, experiment_name: str) -> tuple[int]:
    """Parse lambda1 and lambda2 from a .tif file name and labmda2 range."""
    file_name = file_name.split("_")

    lambda1 = int(file_name[1][:-2])

    lambda2_idx = int(file_name[2][2:])
    lambda2_range = _get_lambda2_range(experiment_name)
    lambda2 = lambda2_range[lambda2_idx]

    return lambda1, lambda2


def _get_lambda2_range(experiment_name: str):
    """Get the lambda2 range from user input using the experiment name."""
    global LAMBDA_2_RANGE, LAMBDA_2_STEP
    ### Parse range
    if isinstance(LAMBDA_2_RANGE, dict):
        if experiment_name not in LAMBDA_2_RANGE:
            raise ValueError(f"{experiment_name} not found in LAMBDA_2_RANGE!")
        LAMBDA_2_RANGE = LAMBDA_2_RANGE[experiment_name]
    assert isinstance(LAMBDA_2_RANGE, (tuple, list))
    lambda_2_idx = LAMBDA_2_RANGE

    ### Parse stepsize
    if isinstance(LAMBDA_2_STEP, dict):
        if experiment_name not in LAMBDA_2_STEP:
            raise ValueError(f"{experiment_name} not found in LAMBDA_2_STEP!")
        LAMBDA_2_STEP = LAMBDA_2_STEP[experiment_name]
    assert isinstance(LAMBDA_2_STEP, (int, float))
    step = LAMBDA_2_STEP

    return np.arange(lambda_2_idx[0], lambda_2_idx[1] + 1, step)


def _process_tif_file(file_path: Path) -> int:
    """Process a .tif file and return the sum of pixel intensities."""
    # print(file_path)
    tif = np.array(Image.open(file_path))
    tif, mask = _normalize_standardize_tif(tif)
    return np.sum(tif * mask)


def _normalize_standardize_tif(tif: np.ndarray) -> np.ndarray:
    """Placeholder function for any image modification steps."""
    mask = _create_hot_pixel_mask(tif)
    # Add other standardization steps here if desired
    return tif, mask


def _create_hot_pixel_mask(tif: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Create a mask to remove hot pixels in the image."""
    if not mask:
        mask = np.ones_like(tif)
    for pixel in PIXELS_TO_SUBTRACT:
        x, y = pixel
        if 0 < x < mask.shape[0] - 1 and 0 < y < mask.shape[1] - 1:
            mask[x - 1 : x + 2, y - 1 : y + 2] = 0
        else:
            print(f"Hot pixel index {pixel} out of bounds, skipping.")
    return mask


def _sort_and_save_data_frame(experiment_name: str, intensities: dict) -> pd.DataFrame:
    """Create and sort df and optionally save it as a .csv file."""
    df = pd.DataFrame.from_dict(intensities, orient="index")
    df.index.name = "Lambda1"
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)

    if not WILL_SAVE_DATA:
        return df
    if OUTPUT_DIR.exists() is False:
        os.mkdir(OUTPUT_DIR)
    print(f"Saving intensities as {experiment_name}.csv...")
    try:
        df.to_csv(f"{OUTPUT_DIR}/{experiment_name}.csv")
    except Exception as e:
        print(f"Error saving {experiment_name}.csv!!!\n{e}")
    return df


def _plot_max_spectra(experiment_name: str, df: pd.DataFrame):
    """Plot the exictation spectra for the max emission and vice versa."""
    # TODO: set colors from wavelengths. Either a single color for max wavelength or a gradient
    if not WILL_PLOT_SPECTRA:
        return
    max_loc = df.stack().idxmax()
    max_lambda1 = df.loc[max_loc[0]]
    max_lambda2 = df[max_loc[1]]

    if PLOT_FORMAT == "subplots":
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(max_lambda1, "o")
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Fluorescence Intensity (a.u.)")
        ax1.set_title(f"{experiment_name}: Lambda 1 spectrum at {max_loc[0]} nm")
        ax1.tick_params(axis="y", which="both", left=False, labelleft=False)

        ax2.plot(max_lambda2, "o")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Fluorescence Intensity (a.u.)")
        ax2.set_title(f"{experiment_name}: Lambda 2 spectrum at {max_loc[1]} nm")
        ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

        plt.tight_layout()
        plt.show()

    elif PLOT_FORMAT == "individual":
        plt.plot(max_lambda1, "o")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Fluorescence Intensity (a.u.)")
        plt.title(f"{experiment_name}: Lambda 1 spectrum at {max_loc[0]} nm")
        plt.tick_params(axis="y", which="both", left=False, labelleft=False)
        plt.show()

        plt.plot(max_lambda2, "o")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Fluorescence Intensity (a.u.)")
        plt.title(f"{experiment_name}: Lambda 2 spectrum at {max_loc[1]} nm")
        plt.tick_params(axis="y", which="both", left=False, labelleft=False)
        plt.show()


stellectrum()
