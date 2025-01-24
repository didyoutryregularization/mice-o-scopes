import configparser
config = configparser.ConfigParser()
config.read("src/config.ini")
from src.u_net import Encoder, Decoder
from src.dataloader import MiceHeartDataset

def train_model():
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()

if __name__ == "__main__":
    train_model()