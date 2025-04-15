import argparse
from config import *
from model import build_model
from train import run_training
from test import run_testing
from dataset.cityscapes import DatasetCityscapesSemantic
from transforms import get_train_transforms, get_eval_transforms
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Cityscapes Semantic Segmentation")

    parser.add_argument("--model", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "unet"],
                        help="Model architecture to use.")
    parser.add_argument("--encoder", type=str, default="efficientnet-b4",
                        help="Encoder name for the backbone.")
    parser.add_argument("--weights", type=str, default="imagenet",
                        help="Pretrained weights for encoder.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "all"],
                        help="Whether to train, test, or both.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = build_model(model_name=args.model, encoder_name=args.encoder, encoder_weights=args.weights)

    train_dataset = DatasetCityscapesSemantic(DATA_PATH, "train", "fine", get_train_transforms(), DEVICE)
    val_dataset = DatasetCityscapesSemantic(DATA_PATH, "val", "fine", get_eval_transforms(), DEVICE)
    test_dataset = DatasetCityscapesSemantic(DATA_PATH, "test", "fine", get_eval_transforms(), DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.__name__ = "cross_entropy"
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    if args.mode in ["train", "all"]:
        run_training(model, train_dataset, val_dataset, optimizer, loss_fn, scheduler, config=globals())

    if args.mode in ["test", "all"]:
        run_testing(model, test_dataset, loss_fn, config=globals())
