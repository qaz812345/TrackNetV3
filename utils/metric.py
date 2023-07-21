import torch

def WBCELoss(y_pred, y, reduce=True):
    """ Weighted Binary Cross Entropy loss function defined in TrackNetV2 paper.

        Args:
            y_pred (torch.Tensor): Predicted values with shape (N, 1, H, W)
            y (torch.Tensor): Ground truth values with shape (N, 1, H, W)
            reduce (bool): Whether to reduce the loss to a single value or not

        Returns:
            (torch.Tensor): Loss value with shape (1,) if reduce, else (N, 1)
    """
    
    loss = (-1)*(torch.square(1 - y_pred) * y * torch.log(torch.clamp(y_pred, 1e-7, 1))\
            + torch.square(y_pred) * (1 - y) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1)))
    if reduce:
        return torch.mean(loss)
    else:
        return torch.mean(torch.flatten(loss, start_dim=1), 1)

def get_metric(TP, TN, FP1, FP2, FN):
    """ Helper function to calculate accuracy, precision, recall, f1, miss_rate

        Args:
            TP (int): Number of true positive samples
            TN (int): Number of true negative samples
            FP1 (int): Number of type 1 false positive samples
            FP2 (int): Number of type 2 false positive samples
            FN (int): Number of false negative samples

        Returns:
            accuracy (float): Accuracy
            precision (float): Precision
            recall (float): Recall
            f1 (float): F1-score
            miss_rate (float): Miss rate
    """

    accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN) if (TP + TN + FP1 + FP2 + FN) > 0 else 0
    precision = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0
    
    return accuracy, precision, recall, f1, miss_rate

