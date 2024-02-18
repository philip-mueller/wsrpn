import random
from dataclasses import dataclass
from typing import List, Optional

import albumentations as A
import cv2
from albumentations.augmentations.crops import functional as ACF
from albumentations.augmentations.geometric import functional as AGF
from omegaconf import MISSING
from torch import nn


@dataclass
class TransformConfig:
    min_visibility: float = MISSING  # Fraction until which to discard a bounding box

    transform_type: str = MISSING  # boxes or labels

    # Resize operations
    val_mode: str = MISSING
    val_size: Optional[List[int]] = MISSING

    train_mode: str = MISSING
    train_size: List[int] = MISSING
    crop_scale_range: List[float] = MISSING
    n_views: int = MISSING

    # Horizontal flip
    random_horizontal_flip: bool = MISSING
    horizontal_flip_prob: Optional[float] = MISSING

    # Random affine transformation
    random_affine: bool = MISSING
    random_affine_prob: Optional[float] = MISSING
    rotation_angle_range: List[float] = MISSING
    translation_fraction_yx: List[float] = MISSING
    scaling_range: List[float] = MISSING

    # Random jitter
    color_jitter: bool = MISSING
    color_jitter_prob: Optional[float] = MISSING
    brightness_jitter_ratio_range: List[float] = MISSING
    contrast_jitter_ratio_range: List[float] = MISSING
    saturation_jitter_ratio_range: List[float] = MISSING
    hue_jitter_ratio_range: List[float] = MISSING

    # Gaussian blurring
    gaussian_blur: bool = MISSING
    gaussian_blur_prob: Optional[float] = MISSING
    gaussian_blur_sigma_range: List[float] = MISSING


def build_transform(config: TransformConfig, mode, pixel_mean, pixel_std) -> nn.Module:
    assert mode in ('train', 'val')
    if mode == 'train':
        augmentations = build_transform_train(config)
    else:
        augmentations = build_transform_val(config)

    normalize = A.Normalize(mean=pixel_mean, std=pixel_std, max_pixel_value=1.)

    bbox_params = A.BboxParams(
        format='coco',
        min_visibility=config.min_visibility,
        label_fields=['labels']
    )

    return A.Compose(
        augmentations + [normalize],
        bbox_params=bbox_params
    )


def build_transform_train(config: TransformConfig) -> List:
    augmentations = []

    # Resize operations
    assert len(config.train_size) == 2
    h, w = config.train_size
    assert config.train_mode in ('resize', 'rect_center', 'rect_random', 'random_crop')
    if config.train_mode == 'random_crop':
        augmentations.append(A.RandomResizedCrop(height=h, width=w, scale=tuple(config.crop_scale_range)))
    else:
        augmentations.append(RESIZE_TRANSFORMS[config.train_mode](height=h, width=w))

    # Horizonral flip
    if config.random_horizontal_flip:
        augmentations.append(A.HorizontalFlip(p=config.horizontal_flip_prob))

    # Random affine transformation
    if config.random_affine:
        augmentations.append(
            A.Affine(scale=config.scaling_range,
                     translate_percent=config.translation_fraction_yx,
                     rotate=config.rotation_angle_range,
                     p=config.random_affine_prob)
        )

    # Color jitter
    if config.color_jitter:
        augmentations.append(
            A.ColorJitter(brightness=tuple(config.brightness_jitter_ratio_range),
                          contrast=tuple(config.contrast_jitter_ratio_range),
                          saturation=tuple(config.saturation_jitter_ratio_range),
                          hue=tuple(config.hue_jitter_ratio_range),
                          p=config.color_jitter_prob)
        )

    # Gaussian blurring
    if config.gaussian_blur:
        augmentations.append(
            A.GaussianBlur(blur_limit=3,
                           sigma_limit=config.gaussian_blur_sigma_range,
                           p=config.gaussian_blur_prob)
        )

    return augmentations


def build_transform_val(config: TransformConfig) -> List:
    # Only resize operations for validation
    val_size = config.val_size if config.val_size is not None else config.train_size
    assert len(val_size) == 2
    h, w = val_size
    assert config.val_mode in ('resize', 'rect_center', 'rect_random')
    return [RESIZE_TRANSFORMS[config.val_mode](height=h, width=w)]


class RectangularizeCenter(A.core.transforms_interface.DualTransform):
    """
    First Center crop to to the short side of the image, then resize.
    Class according to albumentations DualTransform.
    """
    def __init__(self, height: int, width: int, p=1.0):
        super(RectangularizeCenter, self).__init__(p)
        self.height = height
        self.width = width

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        # First center crop
        h, w = img.shape[:2]
        self.short = min(h, w)  # Hope this gets called before apply_to_bbox
        img = ACF.center_crop(img, self.short, self.short)

        # Then resize
        return AGF.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # First center crop
        bbox = ACF.bbox_center_crop(bbox, self.short, self.short, **params)
        # In here, bounding box coordinates are scale invariant, so no resize
        return bbox

    def get_transform_init_args_names(self):
        return ("height", "width")


class RectangularizeRandom(A.core.transforms_interface.DualTransform):
    """
    First random crop to to the short side of the image, then resize.
    Class according to albumentations DualTransform.
    """
    def __init__(self, height: int, width: int, p=1.0):
        super(RectangularizeRandom, self).__init__(always_apply=False, p=p)
        self.height = height
        self.width = width

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        # First random crop
        h, w = img.shape[:2]
        if h >= w:
            # Select random position along x-axis
            rand_pos = random.randint(0, h - w)
            img = ACF.crop(img, x_min=0, y_min=rand_pos, x_max=w, y_max=rand_pos + w)
        else:
            # Select random position along y-axis
            rand_pos = random.randint(0, w - h)
            img = ACF.crop(img, x_min=rand_pos, y_min=0, x_max=rand_pos + h, y_max=h)

        self.rand_pos = rand_pos
        self.h, self.w = h, w

        # Then resize
        return AGF.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_mask(self, img, interpolation=cv2.INTER_NEAREST, **params):
        # First random crop
        rand_pos = self.rand_pos
        h, w = img.shape[:2]
        if h >= w:
            img = ACF.crop(img, x_min=0, y_min=rand_pos, x_max=w, y_max=rand_pos + w)
        else:
            img = ACF.crop(img, x_min=rand_pos, y_min=0, x_max=rand_pos + h, y_max=h)
        # Then resize
        return AGF.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # First random crop
        rand_pos = self.rand_pos
        h, w = self.h, self.w
        if h >= w:
            bbox = ACF.bbox_crop(bbox, x_min=0, y_min=rand_pos, x_max=w, y_max=rand_pos + w, **params)
        else:
            bbox = ACF.bbox_crop(bbox, x_min=rand_pos, y_min=0, x_max=rand_pos + h, y_max=h, **params)
        # In here, bounding box coordinates are scale invariant, so no resize
        return bbox

    def get_transform_init_args_names(self):
        return ("height", "width")


RESIZE_TRANSFORMS = {
    'resize': A.Resize,
    'rect_center': RectangularizeCenter,
    'rect_random': RectangularizeRandom,
}


def plot(image, bboxes=None):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.imshow(image)
    if bboxes is not None:
        for x, y, w, h in bboxes:
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()
