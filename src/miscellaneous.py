import os

import monai.losses as losses
import torch.nn as nn
import torch.optim as optim
import shutil
import PIL


def get_optimizer_class(optimizer_string: str):
    OPTIMIZERS = {"adam": optim.Adam, "sgd": optim.SGD, "rmsprop": optim.RMSprop}

    if optimizer_string not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: {optimizer_string}. Available: {list(OPTIMIZERS.keys())}"
        )

    return OPTIMIZERS[optimizer_string]


def get_loss_function(loss_function_string: str):
    LOSS_FUNCTIONS = {
        "dice": losses.DiceLoss(sigmoid=True),
        "dice_ce": losses.DiceCELoss(sigmoid=True),
        "generalized_dice": losses.GeneralizedDiceLoss(sigmoid=True)
    }

    if loss_function_string not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function: {loss_function_string}. Available: {list(LOSS_FUNCTIONS.keys())}"
        )

    return LOSS_FUNCTIONS[loss_function_string]


def create_folder_structure(base_path="models"):
    """
    Checks the existence of the latest experiment in models/ and creates a new folder for the next experiment.

    Args:
        base_path (str): Base path for the experiments.

    Returns:
        str: Path to the new experiment folder.
    """
    base_path = "models"

    existing_experiments = os.listdir(base_path)
    existing_experiments.sort()

    if existing_experiments:
        latest_experiment = existing_experiments[-1]
        latest_experiment_number = int(latest_experiment.split("_")[-1])
    else:
        latest_experiment_number = 0

    new_experiment_number = latest_experiment_number + 1
    new_experiment_folder = os.path.join(
        base_path, f"experiment_{str(new_experiment_number)}"
    )

    # Create new experiment folder
    os.makedirs(new_experiment_folder)
    # Create subfolder for config.py and experiment.yaml
    os.makedirs(os.path.join(new_experiment_folder, "config"))
    # Create subfolder for model checkpoint
    os.makedirs(os.path.join(new_experiment_folder, "checkpoints"))
    # Create subfolder for predictions
    os.makedirs(os.path.join(new_experiment_folder, "predictions"))
    # Create subfolders for dice and iou predictions
    os.makedirs(os.path.join(new_experiment_folder, "predictions", "dice"))
    os.makedirs(os.path.join(new_experiment_folder, "predictions", "iou"))
    # Create subfolder for metrics
    os.makedirs(os.path.join(new_experiment_folder, "metrics"))

    copy_config_files(new_experiment_folder)

    return new_experiment_folder


def copy_config_files(experiment_folder: str):
    """
    Copy config files to the experiment folder.

    Args:
        experiment_folder (str): Path to the experiment folder.
    """
    # Copy config.py
    shutil.copy("src/config.py", os.path.join(experiment_folder, "config"))

def save_resized_images(image_path: str, output_path:str, resolution: tuple):
    """
    Resize all images in a directory and save them to a new directory.
    Args:
        image_path (str): Path to the directory containing images to resize.
        output_path (str): Path to the directory to save the resized images.
        resolution (tuple): Target resolution as (width, height).
    """

    for file in os.listdir(image_path):
        image = PIL.Image.open(f"{image_path}/{file}")
        image = image.resize(resolution, PIL.Image.LANCZOS)
        image.save(f"{output_path}/{file}")

    print("Images-resizing successfully!")
