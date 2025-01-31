from yacs.config import CfgNode as CN


_C = CN()

_C.TRAINING = CN()
# Number of Epochs
_C.TRAINING.EPOCHS = 100
# Initial Learning Rate for Scheduler
_C.TRAINING.inital_learning_rate = 0.001
# Final Learning Rate for Scheduler
_C.TRAINING.final_learning_rate = 0.00001
# If to use CUdNN Benchmark
_C.TRAINING.cudnn_benchmark = True
# Which loss function to use
_C.TRAINING.loss_function = "dice"  # one out [dice, dicece, ce]
# Optimizer to use
_C.TRAINING.optimizer = "adam"  # Adam, Adamw, RMSprop
# Optimizer hyperparameters
_C.TRAINING.optimizer_hyperparameters = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0}

_C.MODEL = CN()
# Feature sizes of UNet
_C.MODEL.feature_sizes = (3, 64, 128, 256, 512, 1024)


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()