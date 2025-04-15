import torch
import os

PROJECT_NAME = "SegEffNetB4_Cityscapes"
DATA_PATH = "/Workspace/Datasets/Cityscapes"

ENCODER = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 20
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 4
BATCH_SIZE_TEST = 1
CROP_SIZE = 512
MAX_EPOCHS = 60
NUM_WORKERS = 16
LOG_INTERVAL = 5

# Directories
BASE_DIR = f"/Workspace/{PROJECT_NAME}"
LOGS_DIR = os.path.join(BASE_DIR, "Logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "Checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "TestResults")
