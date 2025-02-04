from yacs.config import CfgNode as CN


_C = CN()

_C.TRAINING = CN()
# Number of Epochs
_C.TRAINING.epochs = 500
# If to use CUdNN Benchmark
_C.TRAINING.cudnn_benchmark = True
# Which loss function to use
_C.TRAINING.loss_function = "dice"  # one out [dice, dicece, ce]
# Optimizer to use
_C.TRAINING.optimizer = "adam"  # Adam, Adamw, RMSprop
# Optimizer hyperparameters
_C.TRAINING.learning_rate = 0.0005
# batchsize
_C.TRAINING.batch_size = 4

_C.MODEL = CN()
# Feature sizes of UNet
_C.MODEL.feature_sizes = (3, 64, 128, 256, 512, 1024)

_C.DATA = CN()
# Path to training images
_C.DATA.image_path_train = "data/split/train"
# Path to validation images
_C.DATA.image_path_val = "data/split/val"
# Path to test images
_C.DATA.image_path_test = "data/split/test"
# Resolution of images
_C.DATA.resolution = None  # If not None, resize images to this resolution



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
