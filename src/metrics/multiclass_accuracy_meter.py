from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def get_kth_values(preds: Tensor, ks: Tensor) -> Tensor:
    """Get the kth largest values of a tensor along dim 1.
    Double negation because torch.kthvalue is kth smallest value

    :param preds: (N, C) matrix of predicted probabilities
    :param ks: (N,) number of correct labels per sample
    :return kth_values: (N, 1)
    """
    return -torch.stack([pred.kthvalue(k).values for pred, k in zip(-preds, ks)])[:, None]


class MultiLabelAccuracyMeter(nn.Module):
    def __init__(self, class_names: List[str]):
        super(MultiLabelAccuracyMeter, self).__init__()
        self.class_names = class_names
        self.corrects: Tensor
        self.totals: Tensor
        self.register_buffer('corrects', torch.zeros(len(self.class_names)))
        self.register_buffer('totals', torch.zeros(len(self.class_names)))
        self.reset()

    def reset(self):
        self.corrects.fill_(0.)
        self.totals.fill_(0.)
        self.overall_correct = 0.
        self.overall_total = 0.

    def update(self, preds: Tensor, targets: Tensor):
        """
        For an image with k classes, the k most likely predictions are considered true.

        :param preds: Predicted class probabilities (N x C)
        :param targets: True one hot labels (N, C)
        """
        assert preds.shape == targets.shape

        # Get topk predictions
        ks = targets.sum(1)
        kth_values = get_kth_values(preds, ks)
        preds_hard = (preds >= kth_values).int()

        # Compute correct classificationss
        corrects = preds_hard * targets

        # Add to counts
        self.corrects += corrects.sum(0)
        self.totals += targets.sum(0)
        self.overall_correct += corrects.sum()
        self.overall_total += targets.sum()

    def compute(self):
        res = {}
        for cls, correct, total in zip(self.class_names, self.corrects, self.totals):
            res[cls] = (correct / total).item() if total > 0 else 0.0
        res['overall'] = (self.overall_correct / self.overall_total).item()
        return res


if __name__ == '__main__':
    acc_meter = MultiLabelAccuracyMeter(['a', 'b', 'c', 'd'])
    preds = torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.9, 0.8, 0.1, 0.1]], dtype=torch.float32)
    targets = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.int32)
    acc_meter.update(preds, targets)
    print(acc_meter.compute())
