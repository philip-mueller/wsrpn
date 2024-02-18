import math
from numbers import Number
from typing import Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from einops import repeat
from torchvision.ops import batched_nms

from src.model.model_interface import ObjectDetectorPrediction


def assemble_box_predictions(boxes: Tensor, obj_masks: Tensor, obj_probs: Tensor,
                             class_predictions: Tensor, class_probs: Tensor) \
        -> Tuple[List[Tensor], List[Tensor]]:
    """
    Assembles the predicted box lists (hard predictions and prob predictions) based on the given boxes and predictions.
    :param boxes: (N x ... x 4)
    :param obj_masks: (N x ...) bool mask
    :param obj_probs: (N x ...)
    :param class_predictions: (N x ...) integer ids
    :param class_probs: (N x ... x C)
    :return: box_probs_list, box_prediction_list
    """
    # (N x H_backbone x W_backbone x (5 + C))
    box_probs = torch.cat([boxes,
                           obj_probs.unsqueeze(-1),
                           class_probs], dim=-1)
    # (N x H_backbone x W_backbone x 6)
    box_predictions = torch.cat([boxes,
                                 class_predictions.unsqueeze(-1).to(dtype=boxes.dtype),
                                 obj_probs.unsqueeze(-1)], dim=-1)
    box_probs_list = []
    box_prediction_list = []
    for box_probs_i, box_predictions_i, obj_predictions_i in zip(box_probs, box_predictions, obj_masks):
        box_probs_list.append(box_probs_i[obj_predictions_i, :])  # (M_i x (5 + C))
        box_prediction_list.append(box_predictions_i[obj_predictions_i, :])  # (M_i x 5)
    return box_probs_list, box_prediction_list


def apply_nms(box_probs_list, box_prediction_list, iou_threshold: float):
    box_probs_list_nms = []
    box_prediction_list_nms = []
    for box_probs, box_preds in zip(box_probs_list, box_prediction_list):
        boxes = box_preds[:, 0:4].clone()
        boxes[:, 2:4] = box_preds[:, 0:2] + box_preds[:, 2:4]  # convert box format form x,y,w,h to x1,y1,x2,y2
        cls_idxs = box_preds[:, 4]
        scores = box_preds[:, 5]
        nms_indices = batched_nms(boxes, scores, cls_idxs, iou_threshold=iou_threshold)
        box_probs_list_nms.append(box_probs[nms_indices, :])
        box_prediction_list_nms.append(box_preds[nms_indices, :])
    return box_probs_list_nms, box_prediction_list_nms

def filter_top1_box_per_class(predictions: ObjectDetectorPrediction):
    _, C = predictions.global_prediction_probs.shape
    filtered_box_predictions_hard = []
    filtered_box_predictions_probs = []
    for pred_hard, pred_probs in zip(predictions.box_prediction_hard,
                                     predictions.box_prediction_probs):
        unique_classes = torch.unique(pred_hard[:, 4])
        filtered_box_prediction_hard = []
        filtered_box_prediction_probs = []
        for c in unique_classes:
            # Select boxes of class c
            class_inds = pred_hard[:, 4] == c
            pred_hard_c = pred_hard[class_inds]
            pred_probs_c = pred_probs[class_inds]
            # Select box with highest confidence for class c
            best_idx = pred_hard_c[:, -1].argmax()
            best_pred_hard = pred_hard_c[None, best_idx]
            best_pred_probs = pred_probs_c[None, best_idx]
            # Gather over boxes
            filtered_box_prediction_hard.append(best_pred_hard)
            filtered_box_prediction_probs.append(best_pred_probs)
        # Gather over samples
        filtered_box_predictions_hard.append(torch.cat(filtered_box_prediction_hard) if len(filtered_box_prediction_hard) > 0 else torch.zeros(0, 6))
        filtered_box_predictions_probs.append(torch.cat(filtered_box_prediction_probs) if len(filtered_box_prediction_hard) > 0 else torch.zeros(0, 5 + C))
    # Assign
    predictions.box_prediction_hard = filtered_box_predictions_hard
    predictions.box_prediction_probs = filtered_box_predictions_probs
    return predictions

