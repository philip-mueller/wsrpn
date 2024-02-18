import unittest

import numpy as np
import torch
from mean_average_precision import MeanAveragePrecision2d
from src.metrics.detection_metrics import BBoxMeanAPMetric


PREDICTIONS = [torch.tensor([
    [429, 219, 99, 28, 0, 0.460851],
    [433, 260, 73, 76, 0, 0.269833],
    [518, 314, 85, 55, 0, 0.462608],
    [592, 310, 42, 78, 0, 0.298196],
    [403, 384, 114, 77, 0, 0.382881],
    [405, 429, 114, 41, 0, 0.369369],
    [433, 272, 66, 69, 0, 0.272826],
    [413, 390, 102, 69, 0, 0.619459]
])]

TARGETS = [torch.tensor([
    [439, 157, 117, 84, 0, 0, 0],
    [437, 246, 81, 105, 0, 0, 0],
    [515, 306, 80, 69, 0, 0, 0],
    [407, 386, 124, 90, 0, 0, 0],
    [544, 419, 77, 57, 0, 0, 0],
    [609, 297, 27, 95, 0, 0, 0]
])]


class TestBBoxMeanAPMetric(unittest.TestCase):
    class_names = ["Test class"]
    gt = np.array([
        [439, 157, 556, 241, 0, 0, 0],
        [437, 246, 518, 351, 0, 0, 0],
        [515, 306, 595, 375, 0, 0, 0],
        [407, 386, 531, 476, 0, 0, 0],
        [544, 419, 621, 476, 0, 0, 0],
        [609, 297, 636, 392, 0, 0, 0]
    ])
    preds = np.array([
        [429, 219, 528, 247, 0, 0.460851],
        [433, 260, 506, 336, 0, 0.269833],
        [518, 314, 603, 369, 0, 0.462608],
        [592, 310, 634, 388, 0, 0.298196],
        [403, 384, 517, 461, 0, 0.382881],
        [405, 429, 519, 470, 0, 0.369369],
        [433, 272, 499, 341, 0, 0.272826],
        [413, 390, 515, 459, 0, 0.619459]
    ])

    def test_compute_VOC_PASCAL(self):
        iou_thresholds = [0.5]
        metric = BBoxMeanAPMetric(self.class_names, iou_thresholds, [])
        metric.add(PREDICTIONS, TARGETS)
        results = metric.compute()

        target_metric = MeanAveragePrecision2d(len(self.class_names))
        target_metric.add(self.preds, self.gt)
        target = target_metric.value(iou_thresholds,
                                     recall_thresholds=np.arange(0., 1.01, 0.01),
                                     mpolicy='soft')['mAP']

        self.assertEqual(results['mAP'], target)
        self.assertEqual(results['Test class_mAP'], target)

    def test_compute_COCO(self):
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        metric = BBoxMeanAPMetric(self.class_names, iou_thresholds, [])
        metric.add(PREDICTIONS, TARGETS)
        results = metric.compute()

        target_metric = MeanAveragePrecision2d(len(self.class_names))
        target_metric.add(self.preds, self.gt)
        target = target_metric.value(iou_thresholds,
                                     recall_thresholds=np.arange(0., 1.01, 0.01),
                                     mpolicy='soft')['mAP']

        self.assertEqual(results['mAP'], target)
        self.assertEqual(results['Test class_mAP'], target)


if __name__ == '__main__':
    unittest.main()
