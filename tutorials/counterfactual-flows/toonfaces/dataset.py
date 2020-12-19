import os
from typing import Tuple
from skimage import io
from skimage.transform import resize
from skimage.color import rgba2rgb

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import utils as U

class Toonfaces(Dataset):
    # NOTE(12-09-2020): SETTING 512x512 WILL CPUALLOC 45GB IN MEMORY!
    # NOTE(12-09-2020): _must_ be a power of 2 for flows (maybe padding inclusive?)
    height = 32
    width = 32
    height_no_pad = 32 - 4
    width_no_pad = 32 - 4
    channels = 3  # no longer do RGBA

    def __init__(self, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.attr_to_file: dict = U.load_from_path(root_dir, size=16)
        self.attr_keys = list(self.attr_to_file.keys())
        self.root_dir = root_dir

    def __len__(self):
        return len(self.attr_keys)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        key = self.attr_keys[idx]
        hxw = (Toonfaces.height_no_pad, Toonfaces.width_no_pad)
        img = resize(rgba2rgb(io.imread(self.attr_to_file[key] + ".png")), hxw)
        attrs = U.key_to_attributes(key)
        return dict(image=img, attrs=attrs)

