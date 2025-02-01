from yacs.config import CfgNode as CN


_C = CN()

_C.TRAINING = CN()
# Number of Epochs
<<<<<<< HEAD
_C.TRAINING.EPOCHS = 2
# Initial Learning Rate for Scheduler
_C.TRAINING.inital_learning_rate = 0.001
# Final Learning Rate for Scheduler
_C.TRAINING.final_learning_rate = 0.00001
=======
_C.TRAINING.epochs = 500
>>>>>>> 948aba16d053ea487918193c824d0d0347266235
# If to use CUdNN Benchmark
_C.TRAINING.cudnn_benchmark = True
# Which loss function to use
_C.TRAINING.loss_function = "dice"  # one out [dice, dicece, ce]
# Optimizer to use
_C.TRAINING.optimizer = "adam"  # Adam, Adamw, RMSprop
# Optimizer hyperparameters
<<<<<<< HEAD
_C.TRAINING.learning_rate = 0.001
# Batch size
_C.TRAINING.batch_size = 2
=======
_C.TRAINING.learning_rate = 0.0005
# batchsize
_C.TRAINING.batch_size = 4
>>>>>>> 948aba16d053ea487918193c824d0d0347266235

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
_C.DATA.resolution = 256



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
