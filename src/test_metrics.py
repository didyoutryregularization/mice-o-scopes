import torch


def dice_coef(groundtruth_mask, pred_mask):
    """
    Calculate the Dice coefficient.

    Parameters:
    groundtruth_mask (torch.Tensor): The ground truth binary mask.
    pred_mask (torch.Tensor): The predicted binary mask.
    dice = torch.mean(2 * intersect / total_sum) if total_sum != 0 else torch.tensor(1.0)
    Returns:
    torch.Tensor: The Dice coefficient.
    """
    intersect = torch.sum(pred_mask * groundtruth_mask)
    total_sum = torch.sum(pred_mask) + torch.sum(groundtruth_mask)
    dice = torch.mean(2 * intersect / total_sum) if total_sum != 0 else torch.tensor(1.0)
    return dice


def iou(groundtruth_mask, pred_mask):
    """
    Calculate the Intersection over Union (IoU).

    Parameters:
    groundtruth_mask (torch.Tensor): The ground truth binary mask.
    pred_mask (torch.Tensor): The predicted mask.
    Returns:
    torch.Tensor: The IoU.
    """
    intersect = torch.sum(pred_mask * groundtruth_mask)
    union = torch.sum(pred_mask) + torch.sum(groundtruth_mask) - intersect
    iou = torch.mean(intersect / union)
    return iou
