import torch
from src.u_net import UNet
from src.config import get_cfg_defaults
import os
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt



#cfg = get_cfg_defaults()

model = UNet((3, 64, 128, 256, 512, 1024))
model.cuda()
model.load_state_dict(
    torch.load(f"{experiment_folder}/checkpoints/model_best_dice.pth")
)
model.eval()


image_files = os.listdir("runtime/input_images")

for image_file in image_files:
    image = Image.open(f"runtime/input_images/{image_file}")

    image = ToTensor()(np.array(image))
    original_resolution = image.shape[-2:]
    image = image.unsqueeze(0).cuda()
    
    image = F.interpolate(image, size=(512, 512), mode='area')
    outputs = F.sigmoid(model(image))
    outputs = F.interpolate(outputs, size=original_resolution, mode='nearest-exact')
    outputs = outputs>0.5

    outputs = outputs.squeeze(0).cpu().numpy()
    plt.imshow(outputs[0], cmap="gray")
    plt.grid(False)
    plt.axis("off")
    plt.savefig(f"runtime/predictions/{image_file}", dpi=300)
    plt.close()

    sum_of_outputs = np.sum(outputs)
    metadata_dir = "runtime/metadata"
    os.makedirs(metadata_dir, exist_ok=True)
    with open(f"{metadata_dir}/{image_file.replace("tif", "txt")}", "w") as f:
        f.write(str(sum_of_outputs))
    
    print(f"Prediction for {image_file} saved.")
    