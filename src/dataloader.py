from torch.utils.data import Dataset
import os
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


class MiceHeartDataset(Dataset):
    def __init__(self, image_path_input, image_path_target, transform=None):
        self.image_path_input = image_path_input
        self.image_path_target = image_path_target
        self.image_names = os.listdir(self.image_path_input + "/original")
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_x = Image.open(self.image_path_input + "/original/" + self.image_names[idx])
        image_y = Image.open(self.image_path_target + "/labeled/" + self.image_names[idx])

        image_x = ToTensor()(np.array(image_x))
        image_y = ToTensor()(np.array(image_y, dtype=bool))

        if self.transform:
            image_x = self.transform(image_x)
            image_y = self.transform(image_y)

        return image_x, image_y