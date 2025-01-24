from typing import List
import torch
from src.u_net import Encoder, Decoder
from torch.amp import autocast
import configparser
config = configparser.ConfigParser()
config.read("config.ini")



# Enable CuDNN benchmark for optimized performance
if config["Training"].getboolean("cudnn_benchmark"):
    torch.backends.cudnn.benchmark = True


def train_one_epoch(encoder, decoder, dataloader_train, optimizer, scaler, seg_loss, scheduler=None) -> List[int]:
    """
    Train the encoder and decoder for one epoch.

    Args:
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model.
        dataloader_train (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        seg_loss (callable): Loss function for segmentation.
        scheduler (optional): Learning rate scheduler.

    Returns:
        list: List of training losses for each batch.
    """
    
    loss_hist_train = []

    encoder.train()
    decoder.train()
    for data in dataloader_train:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda().float()
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = decoder(encoder(inputs))
            loss_value = seg_loss(outputs, labels)

        loss_hist_train.append(loss_value.item())

        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()

    if scheduler:
        scheduler.step()

    return loss_hist_train
    

def test_one_epoch(dataloader_test, encoder, decoder, seg_loss) -> List[int]:
    """
    Evaluate the encoder and decoder for one epoch.

    Args:
        dataloader_test (DataLoader): DataLoader for test/validation data.
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model.
        seg_loss (callable): Loss function for segmentation.

    Returns:
        list: List of test losses for each batch.
    """

    loss_hist_test = []

    encoder.eval()
    decoder.eval()    
    with torch.no_grad():
        for data in dataloader_test:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda().float()
            outputs = decoder(encoder(inputs))
            loss_value_test = seg_loss(outputs, labels)
            loss_hist_test.append(loss_value_test.item())

    return loss_hist_test