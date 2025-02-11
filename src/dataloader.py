from torch.utils.data import Dataset
import os
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import random


class MiceHeartDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.image_names = os.listdir(image_path + "/original")
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_x = Image.open(self.image_path + "/original/" + self.image_names[idx])
        image_y = Image.open(self.image_path + "/labeled/" + self.image_names[idx])

        image_x = ToTensor()(np.array(image_x))
        image_y = ToTensor()(np.array(image_y, dtype=bool))

        if self.transform:
            angle = random.choice([0,90,180,270])
            image_x = F.rotate(image_x, angle=angle)
            image_y = F.rotate(image_y, angle=angle)

        return image_x, image_y