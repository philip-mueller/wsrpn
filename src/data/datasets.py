import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image

from src.data import DatasetConfig
from src.data.cxr8 import build_cxr8_dataset
from src.data.data_utils import load_pil_gray, preload
from src.data.transforms import TransformConfig, build_transform
from torch.utils.data import DataLoader, Dataset, default_collate

from src.utils.utils import seed_everything

log = logging.getLogger(__name__)


class WSUP_OD_Dataset(Dataset):
    """
    Dataset for weakly-supervised object detection.
    The dataset will output images and their corresponding annotations.

    :param dataset: Name of the dataset. Currently, only 'cxr8' is implemented.
    :param data_dir: Path to the directory the dataset.
    :param mode: 'train' or 'val'.
    :para prefetch: Prefetch all data at init
    :param transform: Transforms to apply to the images.
    """
    def __init__(self,
                 config: DatasetConfig,
                 mode: str,
                 prefetch: bool = True,
                 transform: Optional[Callable] = None) -> None:
        super().__init__()

        name = config.name

        if name.lower() == 'cxr8':
            self.images, self.annotations, self.class_names = build_cxr8_dataset(config, mode)
            self.load_fn = load_pil_gray
        else:
            raise ValueError(f"Unknown dataset: {name}")

        if prefetch:
            from time import perf_counter
            log.info(f"Prefetching {len(self.images)} images")
            start = perf_counter()
            self.images = preload(
                self.images,
                load_fn=self.load_fn,
            )
            log.info(f'Prefetching images took {perf_counter() - start:.2f}s')

        self.prefetch = prefetch
        self.transform = transform
        self.fallback_transform = self._init_fallback_transform(transform)

    def _init_fallback_transform(self, transform):
        """
        Sometimes a transform crops the only object in the image.
        This function creates a fallback transform that will only resize the image
        """
        bbox_params = None if 'bboxes' not in transform.processors.keys() else A.BboxParams(
            **transform.processors['bboxes'].params.__dict__)
        return None if transform is None else A.Compose(
            [A.Resize(transform[0].height, transform[0].width), *transform[1:]],
            bbox_params=bbox_params,
        )

    def load_annotation(self, index: int) -> np.ndarray:
        annot = np.array(self.annotations[index])
        if annot.ndim == 1:
            bboxes = None
            labels = annot
        else:
            bboxes = annot[:, :4]
            labels = annot[:, 4].astype(int)
        return bboxes, labels

    def transform_sample(
        self,
        image: Image.Image,
        labels: np.ndarray,
        bboxes: Optional[np.ndarray] = None
    ):
        """
        Apply augmentation to the image and labels and optionally to
        the superpixel index maps and bounding boxes if provided.
        """
        if self.transform is not None:
            # Add arguments
            kwargs = {'image': image,
                      'labels': labels}
            kwargs['bboxes'] = bboxes if bboxes is not None else [[0, 0, 1, 1] for _ in range(len(labels))]
            if bboxes is not None:
                kwargs['bboxes'] = bboxes

            # Transform
            transformed = self.transform(**kwargs)
            if len(transformed['labels']) == 0:
                transformed = self.fallback_transform(**kwargs)

            image_transformed = transformed['image']
            labels_transformed = transformed['labels']
            bboxes_transformed = None if bboxes is None else transformed['bboxes']
            return (image_transformed,
                    labels_transformed,
                    bboxes_transformed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        ...


class WSUP_OD_TrainDataset(WSUP_OD_Dataset):
    """
    Returns the image and a one-hot encoded array of all object labels.
    """
    def __init__(self,
                 config: DatasetConfig,
                 prefetch: bool = True,
                 transform: Optional[Callable] = None,
                 n_views: int = 1) -> None:
        super().__init__(
            config=config,
            mode='train',
            prefetch=prefetch,
            transform=transform
        )
        if n_views > 1:
            assert transform is not None
        self.n_views = n_views

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Load image
        image = self.images[index]
        if not self.prefetch:
            image = self.load_fn(image)

        # Convert to numpy and scale to [0, 1]
        image = np.array(image, dtype=np.float32) / 255.

        # each annotation is [x, y, w, h, category_id] or only [category_id]
        bboxes, labels = self.load_annotation(index)
        if bboxes is not None:
            bboxes = np.concatenate([bboxes, labels[:, None]], axis=1)

        image_views = []
        labels_views = []
        bbox_views = []
        for _ in range(self.n_views):
            # Data augmentations
            (image_transformed,
             labels_transformed,
             bboxes_transformed) = self.transform_sample(
                image=image,
                labels=labels,
                bboxes=bboxes
            )

            # Add channel dimension if grayscale
            if image_transformed.ndim == 2:
                image_transformed = image_transformed[:, :, np.newaxis]

            # Convert images to channels first
            image_transformed = np.moveaxis(image_transformed, 2, 0)

            # Convert labels to one-hot
            if len(labels_transformed) == 0:
                labels_transformed = np.zeros(len(self.class_names), dtype=int)
            else:
                labels_transformed = np.eye(len(self.class_names), dtype=int)[labels_transformed]
                labels_transformed = np.max(labels_transformed, axis=0)  # Allocate to one global vector

            image_views.append(image_transformed)
            labels_views.append(labels_transformed)
            bbox_views.append(bboxes_transformed)
        return image_views, labels_views, bbox_views


class WSUP_OD_TestDataset(WSUP_OD_Dataset):
    """
    Returns the image and all bounding boxes.
    """
    def __init__(self,
                 config: DatasetConfig,
                 mode: str = 'val',
                 prefetch: bool = True,
                 transform: Optional[Callable] = None) -> None:
        assert mode in ['val', 'test']
        super().__init__(
            config=config,
            mode=mode,
            prefetch=prefetch,
            transform=transform
        )

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Load image
        image = self.images[index]
        if not self.prefetch:
            image = self.load_fn(image)

        # Convert to numpy and scale to [0, 1]
        image = np.array(image, dtype=np.float32) / 255.

        # each annotation is [x, y, w, h, label]
        bboxes, labels = self.load_annotation(index)

        # Data augmentations
        if self.transform is not None:
            image, labels, bboxes = self.transform_sample(
                image=image,
                labels=labels,
                bboxes=bboxes
            )

        # Add channel dimension if grayscale
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # Convert images to channels first and bboxes and labels back to numpy
        image = np.moveaxis(image, 2, 0)
        bboxes, labels = np.array(bboxes), np.array(labels)

        # Stack bounding boxes and labels back to [x, y, w, h, label]
        bboxes = np.concatenate([bboxes, labels[:, None]], axis=1)

        # Convert labels to one-hot
        global_labels = labels.astype(int)
        global_labels = np.eye(len(self.class_names), dtype=int)[global_labels]
        global_labels = np.max(global_labels, axis=0)  # Allocate to one global vector

        return image, global_labels, bboxes


def build_dataloaders(config: DatasetConfig,
                      pixel_mean: Tuple[float, float, float],
                      pixel_std: Tuple[float, float, float],
                      transform: TransformConfig,
                      batch_size: int,
                      num_workers: int = 0,
                      seed: int = 0,
                      prefetch: bool = True) -> Tuple[Dict[str, DataLoader], List[str]]:
    """
    Builds dataloaders with WSUP_OD datasets.
    """
    # Reproducible data loading
    seed_everything(seed)
    train_ds = WSUP_OD_TrainDataset(config,
                                    prefetch=prefetch,
                                    transform=build_transform(
                                        transform,
                                        mode='train',
                                        pixel_mean=pixel_mean, pixel_std=pixel_std
                                    ),
                                    n_views=transform.n_views)
    val_ds = WSUP_OD_TestDataset(config,
                                 mode='val',
                                 prefetch=prefetch,
                                 transform=build_transform(
                                     transform,
                                     mode='val',
                                     pixel_mean=pixel_mean,
                                     pixel_std=pixel_std
                                 ))
    test_ds = WSUP_OD_TestDataset(config,
                                  mode='test',
                                  prefetch=prefetch,
                                  transform=build_transform(
                                      transform,
                                      mode='val',
                                      pixel_mean=pixel_mean,
                                      pixel_std=pixel_std
                                  ))
    class_names = train_ds.class_names

    # Get one iter of train_ds and val_ds for debugging
    next(iter(train_ds))
    next(iter(val_ds))
    next(iter(test_ds))

    # handle multiple views per sample (for contrastive loss)
    def train_collate_fn(batch: List[Tuple[List[Any], List[Any]]]):
        batch = list(filter(lambda x: x is not None, batch))
        assert len(batch[0]) == 3
        # we just concatenate views ("inner loop"), i.e. different views are
        # next to each other and the next sample is n_view indices apart
        images = [sample[0][view] for sample in batch
                  for view in range(transform.n_views)]
        labels = [sample[1][view] for sample in batch
                  for view in range(transform.n_views)]
        bboxes = [
            torch.tensor(sample[2][view]) if sample[2][view] is not None else None
            for sample in batch for view in range(transform.n_views)
        ]
        return (default_collate(images),
                default_collate(labels),
                bboxes)

    # default_collate does not work with bboxes as they have different sizes
    def test_collate_fn(batch: List[Tuple[Any, ...]]):
        batch = list(filter(lambda x: x is not None, batch))
        assert len(batch[0]) == 3
        images = [sample[0] for sample in batch]
        labels = [sample[1] for sample in batch]
        bboxes = [torch.tensor(sample[2]) for sample in batch]

        return (default_collate(images),
                default_collate(labels),
                bboxes)

    # Create dataloader
    train_loader = DataLoader(train_ds,
                              batch_size // transform.n_views,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers,
                              collate_fn=train_collate_fn)
    val_loader = DataLoader(val_ds,
                            batch_size // transform.n_views,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=test_collate_fn)
    test_loader = DataLoader(test_ds,
                             batch_size // transform.n_views,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=test_collate_fn)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, class_names
