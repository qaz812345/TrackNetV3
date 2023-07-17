import torch
import numpy as np

from utils.general import *


def WBCELoss(y_pred, y, reduce=True):
    # epsilon = 1e-7
    # Weighted Binary Cross Entropy from TrackNetV2
    loss = (-1)*(torch.square(1 - y_pred) * y * torch.log(torch.clamp(y_pred, 1e-7, 1)) + torch.square(y_pred) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1)))
    if reduce:
        return torch.mean(loss)
    else:
        return torch.mean(torch.flatten(loss, start_dim=1), 1)

def get_metric(TP, TN, FP1, FP2, FN):
    """ Helper function Generate input sequences from frames.

        args:
            TP, TN, FP1, FP2, FN - Each float specifying the count for each result type of prediction

        returns:
            accuracy, precision, recall - Each float specifying the value of metric
    """
    accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN) if (TP + TN + FP1 + FP2 + FN) > 0 else 0
    precision = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0
    return accuracy, precision, recall, f1, miss_rate

