from collections import namedtuple
import logging
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from mean_average_precision import MeanAveragePrecision2d
from numpy.typing import ArrayLike
from torchvision.ops import box_iou
import pandas as pd

from src.metrics.rodeo import RoDeO
from tqdm import tqdm

log = logging.getLogger(__name__)


class BBoxMeanAPMetric():
    """
    Compute mean Average Precision for bounding boxes with
    https://github.com/bes-dev/mean_average_precision

    For COCO, select iou_thresholds = np.arange(0.5, 1.0, 0.05)
    For Pascal VOC, select iou_thresholds = 0.5
    """
    def __init__(self, class_names: int, iou_thresholds: ArrayLike = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
                 extra_reported_thresholds: Tuple = (0.5, 0.75)):
        super(BBoxMeanAPMetric, self).__init__()

        self.iou_thresholds = np.array(iou_thresholds)
        self.extra_reported_thresholds = []
        for extra_thres in extra_reported_thresholds:
            found_close = False
            for thres in self.iou_thresholds:
                if np.isclose(thres, extra_thres):
                    self.extra_reported_thresholds.append(thres)
                    found_close = True
            if not found_close:
                raise ValueError(f'{extra_thres} not found in {self.iou_thresholds}')
        self.metric = MeanAveragePrecision2d(len(class_names))
        self.class_names = class_names

    def reset(self):
        self.metric.reset()

    def add(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        Add a batch of predictions and targets to the metric.
        N samples with each M bounding boxes of C classes.

        src.model.model_interface.ObjectDetectorPrediction.box_prediction_hard
        has the correct format for predictions.

        :param predictions: List of N predictions, each a tensor of shape (M x 6)
                            (x, y, w, h, class_id, confidence)
        :param targets: List of N targets, each a tensor of shape (M x 5)
                        (x, y, w, h, class_id)
        """
        for predicted, target in zip(predictions, targets):
            predicted_np = predicted.detach().cpu().numpy().copy()
            target_np = target.detach().cpu().numpy().copy()

            # Convert from [xmin, ymin, w, h, class_id, confidence]
            # to [xmin, ymin, xmax, ymax, class_id, confidence]
            preds = np.zeros((len(predicted_np), 6))
            preds[:, 0:2] = predicted_np[:, :2]
            preds[:, 2:4] = predicted_np[:, :2] + predicted_np[:, 2:4]
            preds[:, 4:6] = predicted_np[:, 4:6]

            # Convert from [xmin, ymin, w, h, class_id]
            # to [xmin, ymin, xmax, ymax, class_id, difficult]
            gt = np.zeros((len(target_np), 7))
            gt[:, 0:2] = target_np[:, :2]
            gt[:, 2:4] = target_np[:, :2] + target_np[:, 2:4]
            gt[:, 4] = target_np[:, 4]

            self.metric.add(preds, gt)

    def compute(self):
        computed_metrics = self.metric.value(iou_thresholds=self.iou_thresholds,
                                             mpolicy="soft",
                                             recall_thresholds=np.arange(0., 1.01, 0.01))
        metrics = {'mAP': computed_metrics['mAP']}

        for c, class_name in enumerate(self.class_names):
            metrics[f'mAP_classes/{class_name}'] = np.mean([computed_metrics[t][c]['ap'] for t in self.iou_thresholds])
            for t in self.extra_reported_thresholds:
                metrics[f'mAP@{t}_classes/{class_name}'] = computed_metrics[t][c]['ap']

        if self.extra_reported_thresholds is not None:
            for t in self.extra_reported_thresholds:
                metrics[f'mAP_thres/mAP@{t}'] = np.mean([computed_metrics[t][c]['ap'] for c in range(len(self.class_names))])

        return metrics


Prediction = namedtuple('Prediction',
                        ['image_index', 'probability', 'coordinates'])


class FrocPerClassMetric():
    """
    See https://github.com/hlk-1135/object-CXR/blob/master/froc.py
    """
    def __init__(self, num_classes: int, fps=(0.125, 0.25, 0.5, 1, 2, 4, 8)):
        super(FrocPerClassMetric, self).__init__()

        self.fps = fps

        self.preds: List[List[Prediction]] = [[] for _ in range(num_classes)]
        self.target_boxes = [[] for _ in range(num_classes)]  # 1 element per sample
        self.num_target_boxes = [0 for _ in range(num_classes)]
        self.image_id = 0
        self.num_classes = num_classes
        self.device = None

    def reset(self):
        self.image_id = 0
        self.preds = [[] for _ in range(self.num_classes)]
        self.target_boxes = [[] for _ in range(self.num_classes)]
        self.num_target_boxes = [0 for _ in range(self.num_classes)]

    def add(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        for predicted, target in zip(predictions, targets):
            self.device = predicted.device
            predicted = predicted.detach().cpu().numpy().copy()
            target = target.detach().cpu().numpy().copy()

            pred_boxes = predicted[:, :4]
            pred_classes = predicted[:, 4]
            pred_probs = predicted[:, 5]
            target_boxes = target[:, :4]
            target_boxes[:, 2:] = target_boxes[:, :2] + target_boxes[:, 2:]  # convert to x1, y1, x2, y2 box
            target_classes = target[:, 4]

            for cls_id in range(self.num_classes):
                cls_target_boxes = [box for box, box_cls in zip(target_boxes, target_classes) if box_cls == cls_id]
                self.target_boxes[cls_id].append(cls_target_boxes)
                self.num_target_boxes[cls_id] += len(cls_target_boxes)
                assert self.target_boxes[cls_id][self.image_id] is cls_target_boxes
            assert sum(len(self.target_boxes[cls_id][self.image_id]) for cls_id in range(self.num_classes)) == len(target_boxes)

            coords = pred_boxes[:, :2] + pred_boxes[:, 2:] / 2  # convert to center coord

            for coord, prob, cls_id in zip(coords, pred_probs, pred_classes):
                self.preds[int(cls_id)].append(Prediction(self.image_id, prob, coord))

            self.image_id += 1

    def compute(self):
        mean_froc = torch.stack([self.compute_cls(cls_id) for cls_id in range(self.num_classes)]).mean()
        return mean_froc

    def compute_cls(self, cls_id):
        num_images = self.image_id
        # sort prediction by probability
        preds = sorted(self.preds[cls_id], key=lambda x: x.probability, reverse=True)

        # compute hits and false positives
        hits = 0
        false_positives = 0
        fps_idx = 0
        object_hitted = set()
        fps = self.fps
        froc = []
        for i in range(len(preds)):
            is_inside = False
            pred = preds[i]
            for box_index, box in enumerate(self.target_boxes[cls_id][pred.image_index]):
                box_id = (pred.image_index, box_index)
                if inside_object(pred, box):
                    is_inside = True
                    if box_id not in object_hitted:
                        hits += 1
                        object_hitted.add(box_id)

            if not is_inside:
                false_positives += 1

            if false_positives / num_images >= fps[fps_idx]:
                sensitivity = hits / self.num_target_boxes[cls_id]
                froc.append(sensitivity)
                fps_idx += 1

                if len(fps) == len(froc):
                    break

        if len(froc) == 0:
            if self.num_target_boxes[cls_id] == 0:
                froc_metric = np.array(1.0)
            else:
                froc_metric = np.array(float(hits) / self.num_target_boxes[cls_id])
        else:
            while len(froc) < len(fps):
                froc.append(froc[-1])
            froc_metric = np.mean(froc)

        return torch.tensor(froc_metric, device=self.device, dtype=torch.float)


def inside_object(pred, box):
    x1, y1, x2, y2 = box
    x, y = pred.coordinates
    return x1 <= x <= x2 and y1 <= y <= y2


class DetectionMetricsWithBootstrap:
    def __init__(self, n_bootstrap: int = 250, **kwargs) -> None:
        self.metrics_kwargs = kwargs
        self.predictions = []
        self.targets = []
        self.n_bootstrap = n_bootstrap

    def add(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def reset(self):
        self.predictions = []
        self.targets = []

    def _compute_metrics(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        metrics = DetectionMetrics(**self.metrics_kwargs)
        metrics.add(predictions, targets)
        return metrics.compute()

    def compute(self, csv_path: str):
        b = len(self.predictions)
        rng = torch.Generator().manual_seed(2147483647)
        idx = torch.arange(b)

        metrics = []
        for _ in tqdm(range(self.n_bootstrap)):
            pred_idx = idx[torch.randint(b, size=(b,), generator=rng)]  # Sample with replacement
            preds = [self.predictions[i] for i in pred_idx]
            targets = [self.targets[i] for i in pred_idx]
            metric_boot = self._compute_metrics(preds, targets)
            metrics.append(metric_boot)

        metric_keys = list(metrics[0].keys())
        bootstarpped_metrics = {
            key: np.stack([metric[key] for metric in metrics])
            for key in metric_keys if not (isinstance(metrics[0][key], np.ndarray) and metrics[0][key].ndim > 0)
        }
        df = pd.DataFrame(bootstarpped_metrics)
        log.info(f'Saving bootstrapped metrics to {csv_path}')
        df.to_csv(csv_path, index=False)
        means = {key: values.mean() for key, values in bootstarpped_metrics.items()}
        stds = {key: values.std() for key, values in bootstarpped_metrics.items()}
        lower = {key: np.quantile(values, 0.025) for key, values in bootstarpped_metrics.items()}
        upper = {key: np.quantile(values, 0.975) for key, values in bootstarpped_metrics.items()}

        metrics = {
            **{f'{key}/mean': mean for key, mean in means.items()},
            **{f'{key}/lower': lower for key, lower in lower.items()},
            **{f'{key}/upper': upper for key, upper in upper.items()},
            **{f'{key}/std': std for key, std in stds.items()}
        }

        return metrics


class DetectionMetrics:
    def __init__(self, class_names, iou_thresholds, extra_reported_thresholds, **kwargs) -> None:
        self.mAP_metric = BBoxMeanAPMetric(
            class_names=class_names,
            iou_thresholds=iou_thresholds,
            extra_reported_thresholds=extra_reported_thresholds
        )
        self.rodeo_metric = RoDeO(class_names=class_names, return_per_class=True)
        self.priou_metric = PRAtIoUMetric(class_names=class_names, obj_thres=None)

        # single box metrics
        self.rodeo_single_box_metric = RoDeO(class_names=class_names, return_per_class=False)
        self.map_single_box_metric = BBoxMeanAPMetric(
            class_names=class_names,
            iou_thresholds=iou_thresholds,
            extra_reported_thresholds=extra_reported_thresholds
        )

        # multi box metrics
        self.rodeo_multi_box_metric = RoDeO(class_names=class_names, return_per_class=False)
        self.map_multi_box_metric = BBoxMeanAPMetric(
            class_names=class_names,
            iou_thresholds=iou_thresholds,
            extra_reported_thresholds=extra_reported_thresholds
        )

    def reset(self):
        self.mAP_metric.reset()
        self.rodeo_metric.reset()
        self.priou_metric.reset()

        self.rodeo_single_box_metric.reset()
        self.map_single_box_metric.reset()
        self.rodeo_multi_box_metric.reset()
        self.map_multi_box_metric.reset()

    def add(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        self.mAP_metric.add(predictions, targets)
        self.priou_metric.add(predictions, targets)
        self.rodeo_metric.add(
            [pred.detach().cpu().numpy() for pred in predictions],
            [target.detach().cpu().numpy() for target in targets]
        )

        preds_single_box, targets_single_box = [], []
        preds_multi_box, targets_multi_box = [], []
        for pred, target in zip(predictions, targets):
            M = target.shape[0]
            assert M > 0
            if M == 1:
                preds_single_box.append(pred)
                targets_single_box.append(target)
            else:
                preds_multi_box.append(pred)
                targets_multi_box.append(target)

        if len(preds_single_box) > 0:
            self.rodeo_single_box_metric.add(
                [pred.detach().cpu().numpy() for pred in preds_single_box],
                [target.detach().cpu().numpy() for target in targets_single_box]
            )
            self.map_single_box_metric.add(preds_single_box, targets_single_box)

        if len(preds_multi_box) > 0:
            self.rodeo_multi_box_metric.add(
                [pred.detach().cpu().numpy() for pred in preds_multi_box],
                [target.detach().cpu().numpy() for target in targets_multi_box]
            )
            self.map_multi_box_metric.add(preds_multi_box, targets_multi_box)

    def compute(self, **kwargs):
        map_metrics = self.mAP_metric.compute()
        rodeo = self.rodeo_metric.compute()
        priou_metric = self.priou_metric.compute()

        map_single_box_metrics = self.map_single_box_metric.compute()
        rodeo_single_box_metrics = self.rodeo_single_box_metric.compute()
        map_multi_box_metrics = self.map_multi_box_metric.compute()
        rodeo_multi_box_metrics = self.rodeo_multi_box_metric.compute()

        single_box_metrics = dict(map_single_box_metrics, **rodeo_single_box_metrics)
        single_box_metrics = {f'single_box/{key}': value for key, value in single_box_metrics.items()}

        multi_box_metrics = dict(map_multi_box_metrics, **rodeo_multi_box_metrics)
        multi_box_metrics = {f'multi_box/{key}': value for key, value in multi_box_metrics.items()}

        return dict(map_metrics, **priou_metric, **rodeo, **single_box_metrics, **multi_box_metrics)


class BoxMetrics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('min_area', torch.tensor(float('inf')))
        self.register_buffer('max_area', torch.tensor(float('-inf')))
        self.mean_areas = []
        self.weighted_mean_areas = []
        self.variance_areas = []

    def reset(self):
        self.mean_areas = []
        self.weighted_mean_areas = []
        self.variance_areas = []
        self.min_area = torch.tensor(float('inf'))
        self.max_area = torch.tensor(float('-inf'))

    def add(self, predictions: List[torch.Tensor]):
        for pred in predictions:
            if pred.shape[0] == 0:
                continue
            pred = pred.detach()
            obj_probs = pred[:, -1]  # (n_rois)
            areas = pred[:, 2] * pred[:, 3]  # (n_rois)

            self.mean_areas.append(areas.mean())
            self.variance_areas.append(areas.var())
            weighted_mean_area = (F.normalize(obj_probs, dim=0, p=1.0) * areas).sum()
            self.weighted_mean_areas.append(weighted_mean_area)
            self.min_area = torch.minimum(self.min_area, areas.min())
            self.max_area = torch.maximum(self.max_area, areas.max())

    def compute(self) -> dict:
        if len(self.mean_areas) == 0:
            return {}
        mean_areas = torch.stack(self.mean_areas)
        variance_areas = torch.stack(self.variance_areas)

        mean_area = mean_areas.mean()
        weighted_mean_area = torch.stack(self.weighted_mean_areas).mean()

        std_area = (mean_areas.var() + variance_areas.mean()).sqrt()
        avg_std_area = variance_areas.sqrt().mean()

        return {
            'roi_area/mean': mean_area.cpu(),
            'roi_area/weighted_mean': weighted_mean_area.cpu(),
            'roi_area/min': self.min_area.cpu(),
            'roi_area/max': self.max_area.cpu(),
            'roi_area/std': std_area.cpu(),
            'roi_area/avg_std': avg_std_area.cpu()
        }


class PRAtIoUMetric(nn.Module):
    def __init__(self, class_names: List[str], thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), obj_thres=None):
        super(PRAtIoUMetric, self).__init__()
        self.class_names = class_names  # note: no-finding should already be excluded
        self.obj_thres = obj_thres
        self.register_buffer('n_tp', torch.zeros(len(class_names), len(thresholds)))  # (C, n_thres)
        self.register_buffer('n_tn', torch.zeros(len(class_names)))  # (C)
        self.register_buffer('n_gt', torch.zeros(len(class_names)))  # (C)
        self.register_buffer('n_pred', torch.zeros(len(class_names)))  # (C)
        self.register_buffer('thresholds', torch.tensor(thresholds))
        self.register_buffer('n', torch.tensor(0))
        self.threshold_values = thresholds

    def reset(self):
        self.n.zero_()
        self.n_tp.zero_()
        self.n_tn.zero_()
        self.n_gt.zero_()
        self.n_pred.zero_()

    def add(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        N = len(predictions)
        C = len(self.class_names)
        self.n += N
        for c in range(C):
            for pred, target in zip(predictions, targets):
                # filter class
                pred = pred[pred[:, 4].long() == c]
                if self.obj_thres is not None:
                    pred = pred[pred[:, 5] >= self.obj_thres]
                target = target[target[:, 4].long() == c]
                n_pred = pred.shape[0]
                n_gt = target.shape[0]
                self.n_pred[c] += n_pred
                self.n_gt[c] += n_gt

                if n_gt == 0 or n_pred == 0:
                    if n_gt == n_pred:  # both zero
                        self.n_tn[c] += 1
                    continue

                # convert x1y1wh -> x1y1x2y2
                pred_boxes = pred[:, :4].clone()
                pred_boxes[:, 2:4] = pred_boxes[:, :2] + pred_boxes[:, 2:4]
                target_boxes = target[:, :4].clone()
                target_boxes[:, 2:4] = target_boxes[:, :2] + target_boxes[:, 2:4]

                ious = box_iou(pred_boxes, target_boxes)  # (n_pred x n_target)
                # select best matching pred box for each target
                ious = ious.amax(0)  # (n_target)
                tp = (ious[:, None] > self.thresholds[None, :]).sum(0)  # (n_thres)
                self.n_tp[c] += tp

    def compute(self):
        tp = self.n_tp.cpu().numpy()
        tn = self.n_tn[:, None].cpu().numpy()
        fp = self.n_pred[:, None].cpu().numpy() - tp
        fn = self.n_gt[:, None].cpu().numpy() - tp

        acc = (tp + tn) / (tp + tn + fp + fn)
        # afp = fp / self.n.cpu().float().numpy()

        metrics = {}

        for t, thres in enumerate(self.threshold_values):
            # for c, class_name in enumerate(self.class_names):
            #     metrics[f'acc_classes/acc@{thres}_{class_name}'] = acc[c, t]
            #     metrics[f'afp_classes/afp@{thres}_{class_name}'] = afp[c, t]

            metrics[f'acc_thres/acc@{thres}'] = np.mean([acc[c, t] for c in range(len(self.class_names))])
            # metrics[f'afp_thres/afp@{thres}'] = np.mean([afp[c, t] for c in range(len(self.class_names))])

        return metrics
