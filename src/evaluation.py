import torch
import statistics
from monai.metrics import compute_iou, DiceMetric
from src.plots import save_image_predictions
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from src.u_net import UNet
import json
import torch.nn.functional as F
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 9331200009



def resize_prediction(prediction, resolution):
    """
    Resize the prediction to the resolution
    """
    prediction_resized =  torch.nn.functional.interpolate(prediction, size=resolution, mode='nearest')
    return prediction_resized!=0  # Binarize the prediction

def compute_evaluation(model, dataloader, resolution, image_predictions_path=False):
    """
    Compute evaluation metric for model.

    Args:
        model (torch.nn.Module): The UNet model.
        dataloader_evaluation (DataLoader): DataLoader for evaluation data.
        image_predictions_path (str): Path to save the image predictions.

    Returns:
        Two ints: Evaluation metrics dice and iou.
    """
    dice_scores = []
    iou_scores = []
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.cuda()
            if image_predictions_path:
                original_inputs = inputs.clone()
            labels = labels.cuda().float()
            if inputs.shape[-1] != resolution:
                print("downsize image")
                inputs = F.interpolate(inputs, size=(resolution, resolution), mode='area')
            outputs = F.sigmoid(model(inputs))
            if outputs.shape[-2:]!=labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='nearest-exact')
            outputs = outputs>0.5
            dice_scores.append(dice_metric(outputs, labels).item())
            iou_scores.append(compute_iou(outputs, labels).item())

            if image_predictions_path:
                # Save image predictions
                save_image_predictions(original_inputs, outputs, labels, f"{image_predictions_path}/{i}.png")

    return statistics.mean(dice_scores), statistics.mean(iou_scores)


def test_best_model(experiment_folder: str, cfg: CfgNode, dataloader: DataLoader, model_mode: str):
    """
    Test the best models. There is model_best_dice.pth and model_best_iou.pth in the checkpoints folder.

    Args:
        experiment_folder (str): Path to the experiment folder to find the best models parameters.
        cfg (CfgNode): Configuration file.
        dataloader (DataLoader): DataLoader for the test set.
        model_metric (str): The metric the model is the best in. Either "dice" or "iou". 
    """
    # Test best model of model metric
    model = UNet(cfg.MODEL.feature_sizes)
    model.cuda()
    model.load_state_dict(
        torch.load(f"{experiment_folder}/checkpoints/model_best_{model_mode}.pth")
    )
    score_test_dice, score_test_iou = compute_evaluation(
        model,
        dataloader,
        resolution=cfg.DATA.resolution,
        image_predictions_path=f"{experiment_folder}/predictions/{model_mode}",
    )

    print(
        f"Best {model_mode} model on test set: Dice: {score_test_dice} IoU: {score_test_iou}"
    )

    # Save dice and iou to metrics folder
    metrics_json = {f"{model_mode}_score_test": score_test_dice if model_mode=="dice" else score_test_iou}
    with open(f"{experiment_folder}/metrics/test_score_{model_mode}.json", "w") as f:
        json.dump(metrics_json, f)