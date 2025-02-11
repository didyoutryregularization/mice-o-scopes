import statistics
from typing import List

import torch
from torch.amp import autocast
import torch.nn.functional as F
from monai.metrics import compute_iou, DiceMetric
from src.plots import save_image_predictions
# from torchmetrics.segmentation import DiceScore, MeanIoU

def train_one_epoch(
    model, dataloader_train, optimizer, scaler, seg_loss, scheduler=None, use_soft_labels=False
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
        if use_soft_labels:
            labels = labels * 0.8 + 0.2  # Adjusts 1 -> 1 and 0 -> 0.2
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
