import statistics
from typing import List

import torch
from torch.amp import autocast
import torch.nn.functional as F
from monai.metrics import compute_iou, DiceMetric
# from torchmetrics.segmentation import DiceScore, MeanIoU
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train_one_epoch(
    model, dataloader_train, optimizer, scaler, seg_loss, scheduler=None
) -> List[int]:
    """
    Train the encoder and decoder for one epoch.

    Args:
        model (torch.nn.Module): The UNet model.
        dataloader_train (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        seg_loss (callable): Loss function for segmentation.
        scheduler (optional): Learning rate scheduler.

    Returns:
        list: List of training losses for each batch.
    """

    loss_hist_train = []

    model.train()
    for data in dataloader_train:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda().float()
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = model(inputs)
            loss_value = seg_loss(outputs, labels)

        loss_hist_train.append(loss_value.item())

        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()

    if scheduler:
        scheduler.step()

    return loss_hist_train


def validate_one_epoch(model, dataloader_val, seg_loss) -> List[int]:
    """
    Validate model for one epoch.

    Args:
        model (torch.nn.Module): The UNet model.
        dataloader_val (DataLoader): DataLoader for validation data.
        seg_loss (callable): Loss function for segmentation.

    Returns:
        list: List of test losses for each batch.
    """

    loss_hist_val = []

    model.eval()
    with torch.no_grad():
        for data in dataloader_val:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda().float()
            outputs = model(inputs)
            loss_value_test = seg_loss(outputs, labels)
            loss_hist_val.append(loss_value_test.item())

    return loss_hist_val


def compute_evaluation(model, dataloader_evaluation, image_predictions_path=False):
    """
    Compute evaluation metric for model.

    Args:
        model (torch.nn.Module): The UNet model. Either "dice" or "iou"
        dataloader_evaluation (DataLoader): DataLoader for evaluation data.
        image_predictions_path (str): Path to save the image predictions.

    Returns:
        int: Evaluation metric.
    """
    dice_scores = []
    iou_scores = []
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader_evaluation):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda().float()
            outputs = F.sigmoid(model(inputs))
            dice_scores.append(dice_metric(outputs>0.5, labels).item())
            iou_scores.append(compute_iou(outputs>0.5, labels).item())

            if image_predictions_path:
                # Save image predictions
                save_image_predictions(inputs, outputs, labels, f"{image_predictions_path}/{i}.png")

    return statistics.mean(dice_scores), statistics.mean(iou_scores)


def save_image_predictions(inputs, outputs, labels, image_predictions_path):
    """
    Save image predictions to disk.

    Args:
        inputs (torch.Tensor): Input images.
        outputs (torch.Tensor): Predicted masks.
        labels (torch.Tensor): Ground truth masks.
        image_predictions_path (str): Path to save the image predictions.
    """
    f, axarr = plt.subplots(1, 3, figsize=(10, 5))
    axarr[0].imshow(inputs[0].cpu().permute(1, 2, 0))
    axarr[1].imshow(labels[0][0].cpu())
    axarr[2].imshow(outputs[0][0].cpu())
    plt.show()
    plt.savefig(image_predictions_path, dpi=300)
