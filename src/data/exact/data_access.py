from .. import data_dir
import os
from .preprocessing import (
    patch_size_mm_to_pixels,
    split_into_patches,
    DEFAULT_PREPROCESS_TRANSFORM,
    split_into_patches_pixels,
)
import numpy as np
from .server import load_by_core_specifier
from enum import Enum, auto


_DATA_DIR = data_dir()
_BACKEND = None
_DATA_CACHE = {}

if "rf_data.npy" in os.listdir(_DATA_DIR):
    _BACKEND = "numpy_big_file"
    _DATA_CACHE["rf_data"] = np.load(
        os.path.join(_DATA_DIR, "rf_data.npy"), mmap_mode="r"
    )
    core_specifiers = np.load(
        os.path.join(_DATA_DIR, "core_specifiers.npy"), allow_pickle=True
    )
    core_specifier2idx = {cs: i for i, cs in enumerate(core_specifiers)}
    _DATA_CACHE["core_specifier2idx"] = core_specifier2idx

else:
    _BACKEND = "dynamic_download"


def get_rf(core_specifier):
    if _BACKEND is None:
        raise ValueError("No data backend found")

    if _BACKEND == "numpy_big_file":
        return _DATA_CACHE["rf_data"][_DATA_CACHE["core_specifier2idx"][core_specifier]]

    else:
        if core_specifier in _DATA_CACHE:
            return _DATA_CACHE[core_specifier]
        elif core_specifier in os.listdir(_DATA_DIR):
            rf = np.load(os.path.join(_DATA_DIR, core_specifier, "image.npy"))
            _DATA_CACHE[core_specifier] = rf
            return rf
        else:
            iq = load_by_core_specifier(core_specifier)
            rf = DEFAULT_PREPROCESS_TRANSFORM(iq)
            np.save(os.path.join(_DATA_DIR, core_specifier, "image.npy"), rf)
            _DATA_CACHE[core_specifier] = rf
            return rf

