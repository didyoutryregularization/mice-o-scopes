import sys
import os
from src.evaluation import test_best_model
# Get the root of the project directory (assuming this script is inside the 'src' directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path (not just the 'src' directory)
sys.path.append(project_root)

import statistics

import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 9331200009

from src.config import get_cfg_defaults
from src.plots import save_loss_plot
from src.dataloader import MiceHeartDataset, custom_collate
from src.miscellaneous import (
    get_loss_function,
    get_optimizer_class,
    create_folder_structure,
)
from src.training_loop import train_one_epoch, validate_one_epoch
from src.evaluation import compute_evaluation
from src.u_net import UNet
import json


def train_model(cfg: CfgNode):
    # Create folder structure for experiment
    experiment_folder = create_folder_structure()

    # Enable CuDNN benchmark for optimized performance
    torch.backends.cudnn.benchmark = cfg.TRAINING.cudnn_benchmark

    model = UNet(cfg.MODEL.feature_sizes)
    model.cuda()

    dataset_train = MiceHeartDataset(
        image_path=cfg.DATA.image_path_train, resolution_inputs=cfg.DATA.resolution_inputs, resolution_outputs=
        cfg.DATA.resolution_outputs
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg.TRAINING.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate
    )

    dataset_val = MiceHeartDataset(
        image_path=cfg.DATA.image_path_val, resolution_inputs=cfg.DATA.resolution, resolution_outputs=
        cfg.DATA.resolution_outputs
    )
    dataloader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate)

    dataset_test = MiceHeartDataset(
        image_path=cfg.DATA.image_path_test, resolution_inputs=cfg.DATA.resolution, resolution_outputs=
        cfg.DATA.resolution_outputs
    )
    dataloader_test = DataLoader(dataset_test, batch_size=1, collate_fn=custom_collate)

    optimizer_class = get_optimizer_class(optimizer_string=cfg.TRAINING.optimizer)
    optimizer = optimizer_class(
        model.parameters(), cfg.TRAINING.learning_rate
    )

    scaler = torch.amp.GradScaler(device="cuda")

    seg_loss = get_loss_function(loss_function_string=cfg.TRAINING.loss_function)

    loss_history_train = []
    loss_history_val = []

    best_iou_score = 0
    best_dice_score = 0

    for i in range(cfg.TRAINING.epochs):
        # Train model
        loss_values_train = train_one_epoch(
            model=model,
            dataloader_train=dataloader_train,
            optimizer=optimizer,
            scaler=scaler,
            seg_loss=seg_loss,
        )
        loss_history_train.append(statistics.mean(loss_values_train))

        loss_values_val = validate_one_epoch(
            model=model, dataloader_val=dataloader_val, seg_loss=seg_loss
        )
        loss_history_val.append(statistics.mean(loss_values_val))

        dice_score, iou_score = compute_evaluation(model, dataloader_val)

        # TODO: put this in a separate function
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            torch.save(
                model.state_dict(),
                f"{experiment_folder}/checkpoints/model_best_dice.pth",
            )

        if iou_score > best_iou_score:
            best_iou_score = iou_score
            torch.save(
                model.state_dict(),
                f"{experiment_folder}/checkpoints/model_best_iou.pth",
            )

        print(
            f"Epoch {i + 1}/{cfg.TRAINING.epochs}: Train loss: {statistics.mean(loss_values_train)}, Validate loss: {statistics.mean(loss_values_val)}, Dice Score: {dice_score}, IoU Score: {iou_score}"
        )

    # Save loss history
    torch.save(
        loss_history_train, f"{experiment_folder}/metrics/loss_history_train.pth"
    )
    torch.save(loss_history_val, f"{experiment_folder}/metrics/loss_history_val.pth")
    # Test models
    test_best_model(
        experiment_folder,
        cfg,
        dataloader_test,
        "dice"
    )
    # Test model
    test_best_model(
        experiment_folder,
        cfg,
        dataloader_test,
        "iou"
    )

    save_loss_plot(save_path=f"{experiment_folder}/metrics/loss_curves.png", train_loss=loss_history_train, val_loss=loss_history_val)



if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("src/experiment.yaml")
    cfg.freeze()
    print(cfg)

    train_model(cfg=cfg)
