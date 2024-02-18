from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_matching(
    preds: np.ndarray,
    targets: np.ndarray,
    shape_weight: Optional[float] = 1.0,
    class_weight: Optional[float] = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Performs hungarian matching on a set of predicted and target boxes
    and returns the matched pairs, as well as the unmatched predicted and
    target boxes.

    Args:
        preds: Array of predicted boxes (M_p x >=4)
        targets: Array of target boxes (M_t x >=4)
        shape_weight: How to weight shape similarity in the cost matrix
        class_weight: How to weight class similarity in the cost matrix
    """
    # Perform matching
    shape_cost = -generalized_box_iou(preds[:, :4], targets[:, :4])
    class_cost = -(preds[:, 4][:, None] == targets[:, 4][None, :]).astype(np.int32)
    cost_matrix = shape_cost * shape_weight + class_cost * class_weight
    pred_inds, target_inds = linear_sum_assignment(cost_matrix)

    # Find matched predictions and targets
    matched_preds = preds[pred_inds]
    matched_targets = targets[target_inds]

    # Find unmatched predictions and targets
    unmatched_preds_mask = np.ones(preds.shape[0], dtype=bool)
    unmatched_preds_mask[pred_inds] = False
    unmatched_preds = preds[unmatched_preds_mask]
    unmatched_targets_mask = np.ones(targets.shape[0], dtype=bool)
    unmatched_targets_mask[target_inds] = False
    unmatched_targets = targets[unmatched_targets_mask]

    return (matched_preds,
            matched_targets,
            unmatched_preds,
            unmatched_targets)


def centered_iou(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    r"""Computes the IoU of pairs of centered boxes (aligned at upper left).

    Args:
        preds: Tensor of predicted boxes (M x 4), each (x,y,w,h)
        targets: Tensor of target boxes (M x 4), each (x,y,w,h)
    """
    preds = preds.copy()
    # Align at upper left
    preds[:, :2] = targets[:, :2]
    # Compute IoUs
    ious = np.diag(box_iou(preds[:, :4], targets[:, :4]))
    return ious


def get_center(boxes: np.ndarray) -> np.ndarray:
    r"""Boxes in (x,y,w,h) format. Returns the center of each box."""
    return boxes[:, :2] + (boxes[:, 2:4] / 2)


def harmonic_mean(x: np.ndarray, eps: Optional[float] = 1e-6) -> np.ndarray:
    return len(x) / (1 / (x + eps)).sum()


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    r"""Compute the intersection over union between all samples in two sets
    of bounding boxes in the (x,y,w,h) format."""
    # Compute the coordinates of the bounding boxes' corners
    x1, y1 = boxes1[:, 0], boxes1[:, 1]
    x2, y2 = x1 + boxes1[:, 2], y1 + boxes1[:, 3]
    x3, y3 = boxes2[:, 0], boxes2[:, 1]
    x4, y4 = x3 + boxes2[:, 2], y3 + boxes2[:, 3]

    # Compute the areas of the bounding boxes
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    # Compute the intersection coordinates and areas
    xA = np.maximum(x1[:, np.newaxis], x3)
    yA = np.maximum(y1[:, np.newaxis], y3)
    xB = np.minimum(x2[:, np.newaxis], x4)
    yB = np.minimum(y2[:, np.newaxis], y4)
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Compute the union areas
    unionArea = area1[:, np.newaxis] + area2 - interArea

    # Compute the intersection over union
    return interArea / unionArea


def generalized_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    r"""Compute the generalized intersection over union between all samples
    in two sets of bounding boxes in the (x,y,w,h) format."""

    # Compute the coordinates of the bounding boxes' corners
    x1, y1 = boxes1[:, 0], boxes1[:, 1]
    x2, y2 = x1 + boxes1[:, 2], y1 + boxes1[:, 3]
    x3, y3 = boxes2[:, 0], boxes2[:, 1]
    x4, y4 = x3 + boxes2[:, 2], y3 + boxes2[:, 3]

    # Compute the areas of the bounding boxes
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    # Compute the intersection coordinates and areas
    xA = np.maximum(x1[:, np.newaxis], x3)
    yA = np.maximum(y1[:, np.newaxis], y3)
    xB = np.minimum(x2[:, np.newaxis], x4)
    yB = np.minimum(y2[:, np.newaxis], y4)
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Compute the union areas
    unionArea = area1[:, np.newaxis] + area2 - interArea

    # Compute the enclosing box
    xC = np.minimum(x1[:, np.newaxis], x3)
    yC = np.minimum(y1[:, np.newaxis], y3)
    xD = np.maximum(x2[:, np.newaxis], x4)
    yD = np.maximum(y2[:, np.newaxis], y4)
    encArea = np.maximum(0, xD - xC) * np.maximum(0, yD - yC)

    # Compute the generalized intersection over union
    return interArea / unionArea - (encArea - unionArea) / encArea
