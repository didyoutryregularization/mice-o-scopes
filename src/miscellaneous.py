import torch.optim as optim
import monai.losses as losses
import torch.nn as nn
import configparser
config = configparser.ConfigParser()
config.read("config.ini")

def get_optimizer_class():
    OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop}

    return OPTIMIZERS[config["Training"]["optimizer"]]

def get_loss_function():
    LOSS_FUNCTIONS = {
        "cross_entropy": nn.BCEWithLogitsLoss,
        "dice": losses.DiceLoss,
        "dice_ce": losses.DiceCELoss
    }

    return LOSS_FUNCTIONS[config["Training"]["loss_function"]]

