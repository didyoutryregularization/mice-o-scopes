import statistics

import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from config import get_cfg_defaults
from src.dataloader import MiceHeartDataset
from src.miscellaneous import get_loss_function, get_optimizer_class
from src.training_loop import train_one_epoch, validate_one_epoch
from src.u_net import UNet


def train_model(cfg: CfgNode):
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

    best_evaluation_metric = torch.inf

    for i in range(cfg.TRAINING.epochs):
        # Train model
        loss_values_train = train_one_epoch(
            model=model,
            dataloader_train=dataloader_train,
            optimizer=optimizer,
            scaler=scaler,
            seg_loss=seg_loss,
        )
        loss_values_val = validate_one_epoch(
            model=model, dataloader_val=dataloader_val, seg_loss=seg_loss
        )
        evaluation_metric = validate_model(model, dataloader_val, seg_loss)  # TODO: implement evaluation metric

        if evaluation_metric < best_evaluation_metric:
            pass  # TODO: implement saving logic

        print(
            f"Epoch {i + 1}/{cfg.TRAINING.epochs}: Train loss: {statistics.mean(loss_values_train)}, Validate loss: {statistics.mean(loss_values_val)}, Evaluation metric: {evaluation_metric}"
        )


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("experiment.yaml")
    cfg.freeze()
    print(cfg)

    train_model(cfg=cfg)
