import statistics

import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from config import get_cfg_defaults
from src.dataloader import MiceHeartDataset
from src.miscellaneous import get_loss_function, get_optimizer_class
from src.training_loop import compute_evaluation, train_one_epoch, validate_one_epoch
from src.u_net import UNet


def train_model(cfg: CfgNode):
    # Enable CuDNN benchmark for optimized performance
    torch.backends.cudnn.benchmark = cfg.TRAINING.cudnn_benchmark

    model = UNet(cfg.TRAINING.feature_sizes)
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

    optimizer_class = get_optimizer_class(optimizer_string=cfg.TRAINING.optimizer)
    optimizer = optimizer_class(
        model.parameters(), **cfg.TRAINING.optimizer_hyperparameters
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

        if dice_score > best_dice_score or iou_score > best_iou_score:
            best_dice_score = dice_score
            best_iou_score = iou_score
            # TODO save model with best scores and all information

        print(
            f"Epoch {i + 1}/{cfg.TRAINING.epochs}: Train loss: {statistics.mean(loss_values_train)}, Validate loss: {statistics.mean(loss_values_val)}, Dice Score: {dice_score}, IoU Score: {iou_score}"
        )


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("experiment.yaml")
    cfg.freeze()
    print(cfg)

    train_model(cfg=cfg)
