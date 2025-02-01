import sys
import os

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
from src.dataloader import MiceHeartDataset
from src.miscellaneous import (
    get_loss_function,
    get_optimizer_class,
    create_folder_structure,
)
from src.training_loop import compute_evaluation, train_one_epoch, validate_one_epoch
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
        image_path=cfg.DATA.image_path_train, resolution=cfg.DATA.resolution
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg.TRAINING.batch_size, shuffle=True, drop_last=True
    )

    dataset_val = MiceHeartDataset(
        image_path=cfg.DATA.image_path_val, resolution=cfg.DATA.resolution
    )
    dataloader_val = DataLoader(dataset_val, batch_size=1)

    dataset_test = MiceHeartDataset(
        image_path=cfg.DATA.image_path_test, resolution=cfg.DATA.resolution
    )
    dataloader_test = DataLoader(dataset_test, batch_size=1)

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


def test_best_model(experiment_folder: str, cfg: CfgNode, dataloader_test: DataLoader, model_metric: str):
    """
    Test the best models on the test set. There is model_best_dice.pth and model_best_iou.pth in the checkpoints folder.

    Args:
        experiment_folder (str): Path to the experiment folder.
    """
    # Test best model of model metric
    model = UNet(cfg.MODEL.feature_sizes)
    model.cuda()
    model.load_state_dict(
        torch.load(f"{experiment_folder}/checkpoints/model_best_{model_metric}.pth")
    )
    score_test, _ = compute_evaluation(
        model,
        dataloader_test,
        image_predictions_path=f"{experiment_folder}/predictions/{model_metric}",
    )

    print(
        f"{model_metric} Score on test set: {score_test}"
    )

    # Save dice and iou to metrics folder
    metrics = {f"{model_metric}_score_test": score_test}
    with open(f"{experiment_folder}/metrics/test_score_{model_metric}.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("src/experiment.yaml")
    cfg.freeze()
    print(cfg)

    train_model(cfg=cfg)
