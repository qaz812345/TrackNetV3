import torch
import numpy as np

from utils.general import *


def predict_location(heatmap, mode='center'):
    """ Get coordinates from the heatmap.

        args:
            heatmap - A numpy.ndarray of a single heatmap with shape (H, W)

        returns:
            ints specifying center coordinates of object
    """
    if np.amax(heatmap) == 0:
        # No respond in heatmap
        if mode == 'center':
            return 0, 0
        if mode == 'bbox':
            return 0, 0, 0, 0
    else:
        # Find all respond area in the heapmap
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # Find largest area amoung all contours
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        target = rects[max_area_idx] # (x, y, w, h)
        if mode == 'center':
            return int((target[0] + target[2] / 2)), int((target[1] + target[3] / 2)) # cx, cy
        if mode == 'bbox':
            '''bbox_list = []
            for i in range(len(rects)):
                bbox_list.append((rects[i][0], rects[i][1], rects[i][0]+rects[i][2], rects[i][1]+rects[i][3])) # x1, y1, x2, y2'''
            return target[0], target[1], target[0]+target[2], target[1]+target[3]

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

def get_confusion_matrix(indices, y_true=None, y_pred=None, c_true=None, c_pred=None, tolerance=4):
    """ Helper function Generate input sequences from frames.

        args:
            indices - A tf.EagerTensor or a list of indices for sequences with shape (N)
            y_true - A tf.EagerTensor of ground-truth heatmap sequences
            y_pred - A tf.EagerTensor of predicted heatmap sequences 
            c_true - A tf.EagerTensor of ground-truth coordinate sequences with shape (N, 2)
            c_pred - A tf.EagerTensor of predicted coordinate sequences with shape (N, 2)
            tolerance - A int speicfying the tolerance for FP1
        returns:
            TP, TN, FP1, FP2, FN - Lists of tuples of all the prediction results
                                    each tuple specifying (sequence_id, frame_no)
    """
    TP, TN, FP1, FP2, FN = 0, 0, 0, 0, 0
    indices = indices.detach().cpu().numpy().tolist() if torch.is_tensor(indices) else indices.numpy().tolist()

    if y_true is not None and y_pred is not None:
        y_pred = y_pred > 0.5
        batch_size, seq_len = y_true.shape[0], y_true.shape[1]
        y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true = to_img_format(y_true) # (N, F, H, W)
        y_pred = to_img_format(y_pred) # (N, F, H, W)
    
    if c_true is not None and c_pred is not None:
        batch_size, seq_len = c_true.shape[0], c_true.shape[1]
        c_true = c_true.detach().cpu().numpy() if torch.is_tensor(c_true) else c_true
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred
        c_true[:, :, 0] = c_true[:, :, 0] * WIDTH
        c_true[:, :, 1] = c_true[:, :, 1] * HEIGHT
        c_pred[:, :, 0] = c_pred[:, :, 0] * WIDTH
        c_pred[:, :, 1] = c_pred[:, :, 1] * HEIGHT

    for n in range(batch_size):
        for f in range(seq_len):
            d_i = indices[n][f]
            if c_true is not None and c_pred is not None:
                c_t = c_true[n][f]
                c_p = c_pred[n][f]
                if np.amax(c_p) == 0 and np.amax(c_t) == 0:
                    # True Negative: prediction is no ball, and ground truth is no ball
                    TN += 1
                elif np.amax(c_p) > 0 and np.amax(c_t) == 0:
                    # False Positive 2: prediction is ball existing, but ground truth is no ball
                    FP2 += 1
                elif np.amax(c_p) == 0 and np.amax(c_t) > 0:
                    # False Negative: prediction is no ball, but ground truth is ball existing
                    FN += 1
                elif np.amax(c_p) > 0 and np.amax(c_t) > 0:
                    # both prediction and ground truth are ball existing
                    cx_pred, cy_pred = int(c_p[0]), int(c_p[1])
                    cx_true, cy_true = int(c_t[0]), int(c_t[1])
                    dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                    if dist > tolerance:
                        # False Positive 1: prediction is ball existing, but is too far from ground truth
                        FP1 += 1
                    else:
                        # True Positive
                        TP += 1
                else:
                    raise ValueError(f'Invalid input: {c_p}, {c_t}')
            elif y_true is not None and y_pred is not None:
                y_t = y_true[n][f]
                y_p = y_pred[n][f]
                
                if np.amax(y_p) == 0 and np.amax(y_t) == 0:
                    # True Negative: prediction is no ball, and ground truth is no ball
                    TN += 1
                elif np.amax(y_p) > 0 and np.amax(y_t) == 0:
                    # False Positive 2: prediction is ball existing, but ground truth is no ball
                    FP2 += 1
                elif np.amax(y_p) == 0 and np.amax(y_t) > 0:
                    # False Negative: prediction is no ball, but ground truth is ball existing
                    FN += 1
                elif np.amax(y_p) > 0 and np.amax(y_t) > 0:
                    # both prediction and ground truth are ball existing
                    # find center coordinate of the contour with max area as prediction
                    cx_true, cy_true = predict_location(to_img(y_t))
                    cx_pred, cy_pred = predict_location(to_img(y_p))
                    dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                    if dist > tolerance:
                        # False Positive 1: prediction is ball existing, but is too far from ground truth
                        FP1 += 1
                    else:
                        # True Positive
                        TP += 1
                else:
                    raise ValueError('Invalid input')
            else:
                raise ValueError('Invalid input')
    return np.array([TP, TN, FP1, FP2, FN])
