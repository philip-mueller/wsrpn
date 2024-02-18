from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection

from src.metrics.detection_metrics import (BoxMetrics, DetectionMetrics, DetectionMetricsWithBootstrap)
from src.metrics.multiclass_accuracy_meter import MultiLabelAccuracyMeter


def build_metrics(class_names, bootstrap=False):
    detection_args = {
        'class_names': class_names,
        'iou_thresholds': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        'extra_reported_thresholds': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    }
    if bootstrap:
        detection_metrics = DetectionMetricsWithBootstrap(**detection_args)
    else:
        detection_metrics = DetectionMetrics(**detection_args)
    
    classification_metrics = MetricCollection({
        'acc': Accuracy(num_classes=1, multiclass=False, average='samples',
                        subset_accuracy=True),
        'f1': F1Score(num_classes=1, multiclass=False, average='samples')
    })
    accuracy_meter = MultiLabelAccuracyMeter(class_names)
    auroc_metrics = {
        'global': AUROC(num_classes=len(class_names)),
        'patch_aggregated': AUROC(num_classes=len(class_names)),
        'roi_aggregated': AUROC(num_classes=len(class_names))
    }
    box_metrics = BoxMetrics()
    return (detection_metrics, classification_metrics, auroc_metrics,
            accuracy_meter, box_metrics)
