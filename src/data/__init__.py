from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    name: str = MISSING
    data_dir: str = MISSING

    train_val_split: Optional[List[float]] = MISSING

    # Normalization parameters
    pixel_mean: List[float] = MISSING
    pixel_std: List[float] = MISSING
