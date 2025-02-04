# MICE-O-SCOPES

This repository contains several approaches to segment scar tissue in microscopic images of mice hearts using deep learning (Binary Image Segmentation). This makes it possible to determine in a downstream task whether medication leads to milder heart attacks. 

This project comes with a number of challenges. One challenge is the limited dataset, consisting of only 100 images with corresponding segmentations. Additionally, the images have extremely high and varying resolutions. The resolution of the images ranges from 5,000x3,500 to 15,000x15,000 pixels. Due to the enormous size, it is not possible to fully segment an image in its original resolution. On the other hand, the method presented must be able to handle the different resolutions.

The first approach is to train a vanilla U-Net model on resized images to a quadratical shape, e.g. 256x256. After generating predictions at the resized resolution, the predicted masks are upscaled back to their original dimensions (e.g., 15,000x15,000 pixels) before computing evaluation metrics such as Dice and IoU. The original sized prediction is then used to compute the metrics, i.e. Dice and IoU.

|Model|Loss|Learning Rate|Resolution|Augmentation|Dice|IoU|
|---|---|---|---|---|---|---|
|U-Net|Dice|0.0005|256|x|TBD|TBD|
|U-Net|DiceCE|0.0005|256|x|TBD|TBD|
|U-Net|GeneralizedDice|0.0005|256|x|TBD|TBD|