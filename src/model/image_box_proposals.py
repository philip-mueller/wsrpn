import os
import urllib.request
from glob import glob
from os.path import dirname

import cv2
import numpy as np
import torch
from torch import Tensor

from src.settings import RESOURCE_DIR

this_dir = dirname(os.path.realpath(__file__))
edge_box_model_dir = RESOURCE_DIR
edge_box_model_url = "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz"
edge_box_model_file = os.path.join(edge_box_model_dir, "edge_box_model.yml.gz")

# Make faster
cv2.setUseOptimized(True)
cv2.setNumThreads(4)


def selective_search(img: Tensor, mode: str = 'fast') -> Tensor:
    """Find bounding box proposals with selective search

    :param img: (C, H, W) Input image, float32
    :param mode: 'fast' or 'quality'
    :return boxes: (M, 4) Bounding box proposals with (x1, y1, x2, y2)
    """
    img_proc = img.clone()
    # Normalize to [0, 1]
    img_proc = (img_proc - img_proc.min()) / (img_proc.max() - img_proc.min())
    # Convert to (H, W, C) and uint8
    img_proc = np.uint8(img_proc.permute(1, 2, 0) * 255)

    # Expand to 3 channels if necessary
    if img_proc.shape[2] == 1:
        img_proc = np.repeat(img_proc, 3, axis=2)

    # Set up algorithm
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_proc)
    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    elif mode == 'quality':
        ss.switchToSelectiveSearchQuality()
    else:
        raise ValueError(f"Unknown mode {mode}. Select 'fast' or 'quality'.")

    # Process
    boxes = torch.tensor(ss.process(), dtype=torch.float32)

    # Make x, y, h, and w relative to the image.
    C, W, H = img.shape
    boxes[:, (0, 2)] /= W
    boxes[:, (1, 3)] /= H

    # Convert to (x1, y1, x2, y2)
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]

    return boxes


def download_edge_box_model():
    print(
        f"Downloading model for Edge Box algorithm from {edge_box_model_url}",
        f"to {edge_box_model_file}"
    )
    os.makedirs(edge_box_model_dir, exist_ok=True)
    urllib.request.urlretrieve(edge_box_model_url, edge_box_model_file)


def edge_boxes(img: Tensor) -> Tensor:
    """Find bounding box proposals in an image using the Edge Box algorithm

    :param img: (C, H, W) Input image, float32
    :return boxes: (M, 4) Bounding box proposals with (x1, y1, x2, y2)
    """
    # Download edge box model
    if not os.path.exists(edge_box_model_file):
        download_edge_box_model()
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(
        edge_box_model_file
    )

    # Convert to (H, W, C) and numpy
    img_proc = img.permute(1, 2, 0).numpy()
    # Normalize to [0, 1]
    img_proc = (img_proc - img_proc.min()) / (img_proc.max() - img_proc.min())

    # Expand to 3 channels if necessary
    if img_proc.shape[2] == 1:
        img_proc = np.repeat(img_proc, 3, axis=2)

    # Process
    edges = edge_detection.detectEdges(img_proc)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)[0]
    boxes = torch.tensor(boxes, dtype=torch.float32)

    # Make x, y, h, and w relative to the image.
    C, W, H = img.shape
    boxes[:, (0, 2)] /= W
    boxes[:, (1, 3)] /= H

    # Convert to (x1, y1, x2, y2)
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]

    return boxes


def random_boxes(M: int) -> Tensor:
    """Generate M random bounding boxes.

    :param N: Number of bounding boxes to create
    :return boxes: (M, 4) Bounding box proposals with (x1, y1, x2, y2)
    """
    # Uniformly select center locations
    center = torch.rand(M, 2)
    # Find shortest distance to image boundary in x and y direction
    max_wh = torch.stack([center, 1 - center]).min(dim=0).values * 2
    # Uniformly select width and height
    wh = torch.rand(M, 2) * max_wh
    # Get x and y position from center, and width and height
    xy1 = center - (wh / 2)
    xy2 = center + (wh / 2)
    # Build boxes
    boxes = torch.cat([xy1, xy2], dim=1)
    return boxes


def draw_boxes(img: Tensor, boxes: Tensor) -> np.ndarray:
    """
    Draw bounding boxes onto an image.

    :param img: (H, W) Grayscale image, float32 [0, 1]
    :param boxes: (M, 4) Bounding boxes (x1, y1, x2, y2), relative to image size
    """
    img_out = np.uint8(img * 255)
    boxes = boxes.clone()
    boxes[:, (0, 2)] *= img.shape[0]
    boxes[:, (1, 3)] *= img.shape[1]
    boxes = boxes.int()
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
    return img_out


if __name__ == '__main__':
    filename = glob("/datasets/CXR14/images/*.png")[0]
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # (H, W)
    img = cv2.resize(img, (224, 224))  # (H, W)
    img = torch.tensor(img / 255., dtype=torch.float32)
    img_rgb = img[None].repeat(3, 1, 1)  # (C, H, W)

    from time import perf_counter
    t_start = perf_counter()
    boxes_rn = random_boxes(200)
    print(f"Random boxes: {perf_counter() - t_start:.2f}s")
    print(boxes_rn.shape)

    t_start = perf_counter()
    boxes_ss = selective_search(img_rgb, mode='quality')
    print(f"Selective search: {perf_counter() - t_start:.2f}s")
    print(boxes_ss.shape)

    t_start = perf_counter()
    boxes_eb = edge_boxes(img_rgb)
    print(f"Edge boxes: {perf_counter() - t_start:.2f}s")
    print(boxes_eb.shape)

    img_show_rn = draw_boxes(img, boxes_rn)
    img_show_ss = draw_boxes(img, boxes_ss)
    img_show_eb = draw_boxes(img, boxes_eb)
    i = 1
