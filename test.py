import torch
from torch.utils.data import DataLoader
from epoch_test import TestEpoch
from glob import glob
import os

def run_testing(model, test_ds, criterion, config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE_TEST, shuffle=False, num_workers=config.NUM_WORKERS)

    # Load latest/best checkpoint
    best_model_path = sorted(glob(os.path.join(config.CHECKPOINT_DIR, "*.pth")))[-1]
    print(f"Loading model: {best_model_path}")
    model = torch.load(best_model_path)

    tester = TestEpoch(model, criterion, p_dir_export=config.OUTPUT_DIR, device=config.DEVICE, verbose=True)
    tester.run(test_loader)
