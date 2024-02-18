import os
from multiprocessing import Pool
from typing import Callable, Sequence, List

from PIL import Image


def load_pil_rgb(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def load_pil_gray(path: str) -> Image.Image:
    return Image.open(path).convert('L')


def preload(files: Sequence, load_fn: Callable = load_pil_rgb,
            num_processes: int = min(12, os.cpu_count())) -> List:
    """
    Multiprocessing to load all files to RAM fast.

    :param files: List of file paths.
    :param load_fn: Function to load the image.
    :param num_processes: Number of processes to use.
    """
    with Pool(num_processes) as pool:
        results = pool.map(load_fn, files)
    return results
