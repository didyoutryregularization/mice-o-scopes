# MICE-O-SCOPES

This repository explores deep learning approaches for segmenting scar tissue in microscopic images of mice hearts (Binary Image Segmentation). The goal of this research is to assess whether medication can mitigate heart attack severity by analyzing tissue damage.

## Challenges
This project presents several challenges:
- **Limited Data**: Only 100 images with corresponding segmentation masks are available, making training deep learning models more difficult.
- **High-Resolution Images**: The images range in size from **5,000×3,500 to 15,000×15,000 pixels**, making full-resolution segmentation infeasible due to memory constraints.
- **Varying Resolutions**: The images have different resolutions, but the chosen approach requires a fixed input size. This necessitates resizing, which can lead to a loss of detail, or alternative strategies such as patch-based processing.

## Methodology
The initial approach involves training a **vanilla U-Net** on downsampled images resized to a fixed quadratic shape (e.g., **256×256**). At inference, the predicted mask is upsampled back to its original resolution (e.g., **15,000×15,000 pixels**) before computing evaluation metrics such as **Dice coefficient** and **Intersection over Union (IoU)**.

## Results
| Model | Loss Function | Learning Rate | Input Resolution | Data Augmentation | Dice | IoU |
|-------|-------------|--------------|----------------|----------------|------|----|
| U-Net | Dice | 0.0005 | 256×256 | No | TBD | TBD |
| U-Net | DiceCE | 0.0005 | 256×256 | No | TBD | TBD |
| U-Net | GeneralizedDice | 0.0005 | 256×256 | No | TBD | TBD |

(*TBD: Results pending evaluation.*)
