import configparser
from src.u_net import UNet
from src.training_loop import train_one_epoch, test_one_epoch
from src.dataloader import MiceHeartDataset
from src.miscellaneous import get_optimizer_class
import torch
from torch.utils.data import DataLoader

config = configparser.ConfigParser()
config.read("config.ini")


def train_model():
    model = UNet(feature_sizes=config["Model"]["feature_sizes"])
    model.cuda()

    dataset_train = MiceHeartDataset(image_path=config["Data"]["image_path"], resolution=config["Data"].getint("resolution"))
    dataloader_train = DataLoader(dataset_train, batch_size=config["Training"].getint("batch_size"), shuffle=True, drop_last=True)
    optimizer_class = get_optimizer_class()
    optimizer = optimizer_class(model.parameters(), **dict(config["Training"]["optimizer_hyperparameters"]))
    scaler = torch.cuda.amp.GradScaler()
    
    for i in range(config["Training"].getint("epochs")):
        # Train model
        loss_values = train_one_epoch()

if __name__ == "__main__":
    train_model()