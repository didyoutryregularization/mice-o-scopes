import torch.optim as optim
import monai.losses as losses
import torch.nn as nn


def get_optimizer_class(optimizer_string: str):
    OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop}

    if optimizer_string not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_string}. Available: {list(OPTIMIZERS.keys())}")


    return OPTIMIZERS[optimizer_string]

def get_loss_function(loss_function_string: str):
    LOSS_FUNCTIONS = {
        "cross_entropy": nn.BCEWithLogitsLoss,
        "dice": losses.DiceLoss,
        "dice_ce": losses.DiceCELoss
    }
    
    if loss_function_string not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {loss_function_string}. Available: {list(LOSS_FUNCTIONS.keys())}")
    
    return LOSS_FUNCTIONS[loss_function_string]

