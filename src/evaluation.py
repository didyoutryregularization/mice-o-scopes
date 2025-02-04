import torch
from src.

def resize_prediction(prediction, resolution):
    """
    Resize the prediction to the resolution
    """
    prediction_resized =  torch.nn.functional.interpolate(prediction, size=resolution, mode='nearest')
    return prediction_resized!=0  # Binarize the prediction

def evaluate_original_size(dataloader_evaluation):
    """
    Evaluate the prediction at the original size
    """
