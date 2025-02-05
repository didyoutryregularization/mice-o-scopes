import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2

class Block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BatchNorm after conv1
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)  # BatchNorm after conv2

    def forward(self, x):
        # Apply Conv -> BatchNorm -> ReLU in sequence for each layer
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, feature_sizes):
        super().__init__()
        self.feature_sizes = feature_sizes
        self.encoder_blocks = nn.ModuleList(
            [
                Block(feature_sizes[i], feature_sizes[i + 1])
                for i in range(len(feature_sizes) - 1)
            ]
        )
        self.down_pooling = nn.MaxPool2d(2)

    def forward(self, x):
        encoder_features = []
        for i in range(len(self.feature_sizes) - 1):
            x = self.encoder_blocks[i](x)
            encoder_features.append(x)
            x = self.down_pooling(x)

        return encoder_features


class Decoder(nn.Module):
    def __init__(self, feature_sizes=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.feature_sizes = feature_sizes
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(feature_sizes[i], feature_sizes[i + 1], 3, 2, 1, 1)
                for i in range(len(feature_sizes) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                Block(feature_sizes[i], feature_sizes[i + 1])
                for i in range(len(feature_sizes) - 1)
            ]
        )
        self.final_block = Block(feature_sizes[-1], 3)

    def forward(self, encoder_features):
        x = encoder_features[-1]
        for i in range(len(self.feature_sizes) - 1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[-i - 2]], dim=1)
            x = self.dec_blocks[i](x)
        x = self.final_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, feature_sizes=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoder = Encoder(feature_sizes=feature_sizes)
        self.decoder = Decoder(
            feature_sizes=feature_sizes[::-1][:-1]
        )  # Reverse and skip last
        self.final_block = Block(6, 1)
        self.final_layer = nn.Conv2d(1, 1, 3, padding=1)  # So that final layer is not relu

    def forward(self, x: torch.tensor, resolution: tuple):
        original_image = x.clone()
        print(f"resolution of x: {resolution}")
        x = self.decoder(self.encoder(x))
        x = F.interpolate(x, size=resolution, mode="bilinear", align_corners=False)
        print(f"Shape of x after decoder incl. interpolation: {x.shape}")
        x = torch.cat([original_image, x], dim=1)
        print(f"Shape of x after cat: {x.shape}")
        x = self.final_block(x)
        print(f"Shape of x after final block: {x.shape}")
        x = self.final_layer(x)
        print(f"Shape of x after final layer: {x.shape}")
        return x



