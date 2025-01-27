from torch.utils.data import Dataset, DataLoader
import torch
import os
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np


class MiceHeartDataset(Dataset):
    def __init__(self, image_path, resolution = 256, transform=None):
        self.image_path = image_path
        self.image_names = os.listdir(image_path+"/original")
        self.transform = transform
        self.resolution = resolution

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_x = ToTensor()((np.array(Image.open(self.image_path + "/original/" + self.image_names[idx]).resize((self.resolution,self.resolution)))))/255
        image_y = ToTensor()((np.array(Image.open(self.image_path + "/labeled/" + self.image_names[idx]).resize((self.resolution,self.resolution)), dtype=bool)))

        if self.transform:
            image_x = self.transform(image_x)
            image_y = self.transform(image_y)
    
        return image_x, image_y