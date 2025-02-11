import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def save_loss_plot(save_path, train_loss, val_loss):
    """
    Saves a loss plot with given training and validation loss lists.
    
    Parameters:
        save_path (str): Path to save the plot (e.g., 'loss_plot.png').
        train_loss (list): List of training loss values.
        val_loss (list): List of validation loss values.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)  # Enable grid

    plt.savefig(save_path, dpi=300)  # Save the figure
    plt.close()  # Close the plot to free memory


def save_image_predictions(inputs, outputs, labels, image_predictions_path: str):
    """
    Save image predictions to disk.

    Args:
        inputs (torch.Tensor): Input images.
        outputs (torch.Tensor): Predicted masks.
        labels (torch.Tensor): Ground truth masks.
        image_predictions_path (str): Path to save the image predictions.
    """
    f, axarr = plt.subplots(1, 3, figsize=(10, 5))
    axarr[0].imshow(inputs[0].cpu().permute(1, 2, 0))
    axarr[1].imshow(labels[0][0].cpu())
    axarr[2].imshow(outputs[0][0].cpu()>0.5)
    plt.savefig(image_predictions_path, dpi=300)
    plt.close()  # Close the plot to free memory
    plt.imshow(outputs[0][0].cpu()>0.5, cmap="grey")
    plt.savefig(image_predictions_path.replace(".png", "_prediction.png"), dpi=300)
