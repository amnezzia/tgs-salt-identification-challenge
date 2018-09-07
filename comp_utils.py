"""
Various helpers and metric functions
"""
import numpy as np
import torch
import torch.nn.functional as F

import cv2
import os
import json
import pickle


def iou(pred, target, th=0.5):
    """
    get per image IOU but it should return 1 if both pred and target are empty

    Parameters
    ----------
    pred    pytorch tensor with predicted masks
    target  pytorch tensor with target masks
    th      threshold to use for predicted masks

    Returns
    -------
        iou metric per image
    """
    n = pred.size(0)
    pred = (pred > th)
    target = (target > 0.5)
    intersection = (target & pred).long()
    union = (target | pred).long()
    return (intersection.view(n, -1).sum(dim=1).float() + 1e-3) / (union.view(n, -1).sum(dim=1).float() + 1e-3)


def ap_iou(pred, target, th=0.5):
    """
    competition specific average precision IOU

    Parameters
    ----------
    pred    pytorch tensor with predicted masks
    target  pytorch tensor with target masks
    th      threshold to use for predicted masks

    Returns
    -------
        total metric value for the dataset
    """
    per_sample_iou = iou(pred, target, th=th)

    ths = torch.range(0.5, 1, 0.05).to(pred.device)[None, :]
    hits = (per_sample_iou[:, None] > ths).to(torch.float32)

    return hits.mean()


def ap_iou_with_logits(pred, target, th=0.5):
    """
    adapter for ap_iou in case when sigmoid activation was not yet applied to the predicted masks
    """
    pred = F.sigmoid(pred)
    return ap_iou(pred, target, th=th)


def dice_loss_logits(pred, target):
    """
    Parameters
    ----------
    pred    pytorch tensor with predicted masks (logits)
    target  pytorch tensor with target masks

    Returns
    -------
        dice loss per image, no averaging over batch
    """
    pred = F.sigmoid(pred)
    return dice_loss(pred, target)


def dice_loss(pred, target):
    """
    Parameters
    ----------
    pred    pytorch tensor with predicted masks
    target  pytorch tensor with target masks

    Returns
    -------
        dice loss per image, no averaging over batch
    """
    smooth = 0.1
    n = pred.size(0)
    intersection = (pred * target).view(n, -1).sum(dim=-1) + smooth
    union = (pred + target).view(n, -1).sum(dim=-1) + smooth
    return intersection / union


def load_img(img_id, train=True):
    """Given image ID, load an image with a mask (or just an image) if train (not train)"""
    if train:
        img = cv2.imread('train_images/{}.png'.format(img_id)).max(axis=-1, keepdims=1)
        mask = cv2.imread('train_masks/{}.png'.format(img_id)).max(axis=-1, keepdims=1)
        img = np.concatenate([img, mask], axis=-1)
    else:
        img = cv2.imread('test_images/{}.png'.format(img_id)).max(axis=-1, keepdims=1)
    return img


def rle_to_mask(rle_str, out_shape=(101, 101)):
    """
    Helper method to convert rle mask to an image mask
    """
    out = np.zeros(np.prod(out_shape))

    rle = np.array(rle_str.split()).astype(int)
    for ix, rl in zip(rle[::2], rle[1::2]):
        out[ix:ix + rl] = 1

    return out.reshape(out_shape).transpose()


def make_submission(test_df, predictions, fpath, threshold=0.5, mean_type='simple'):

    if isinstance(predictions, list):
        predictions = np.concatenate([pred[None, ...] for pred in predictions])
        if mean_type == 'simple':
            predictions = predictions.mean(axis=0)
        elif mean_type == 'g':
            predictions = np.exp(np.log(predictions) + 0.01).mean(axis=0)
        elif isinstance(mean_type, (int, float)):
            predictions = ((predictions**mean_type)**(1/mean_type)).mean(axis=0)

    predictions = predictions > threshold
    predictions = predictions.transpose(0, 2, 1)

    n = predictions.shape[0]
    predictions = predictions.reshape((n, -1))

    masks_diffs = np.diff(np.concatenate([np.zeros((n, 1)), predictions, np.zeros((n, 1))], axis=1), axis=1)

    test_df['rle_mask'] = [to_rle(md) for md in masks_diffs]
    test_df.to_csv(fpath, index=False)


def to_rle(mask_diff):
    """Helper function for making rle masks out of an array"""
    rle = np.where(mask_diff != 0)[0] + 1
    if len(rle):
        rle[1::2] = np.diff(rle)[0::2]
    return ' '.join(rle.astype(str))


def final_val_score(experiment_dir, targets_np, metric_fn, splits):
    all_preds = []
    for fold_i in range(len(splits)):
        with open(os.path.join(experiment_dir, 'train_predictions_{}.pickle'.format(fold_i)), 'rb') as f:
            train_predictions = pickle.load(f)
            all_preds.append(train_predictions)

    val_predictions = np.empty_like(all_preds[0])
    for fold_i, (_, val_ixs) in enumerate(splits):
        val_predictions[val_ixs] = all_preds[fold_i][val_ixs]

    res = metric_fn(val_predictions, targets_np)

    with open(os.path.join(experiment_dir, 'metadata.json')) as f:
        meta = json.load(f)

    meta['final_val_score'] = res

    with open(os.path.join(experiment_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)

    return res


class BCE_LogDice(object):
    """
    Loss function that combines BCE and dice losses with equal weights
    """
    __name__ = 'BCE_LogDice'

    def __init__(self, with_logits=True):
        self.with_logits = with_logits

    def __call__(self, pred, target):
        if self.with_logits:
            return self._call_with_logits(pred, target)
        else:
            return self._call(pred, target)

    def _call_with_logits(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        log_dice = torch.log(dice_loss_logits(pred, target)).mean()
        return bce - log_dice

    def _call(self, pred, target):
        bce = F.binary_cross_entropy(pred, target)
        log_dice = torch.log(dice_loss(pred, target)).mean()
        return bce - log_dice
