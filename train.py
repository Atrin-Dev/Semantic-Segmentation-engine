from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from epoch_train import TrainEpoch
from epoch_val import ValidEpoch
import torch
import os

def run_training(model, train_ds, val_ds, optimizer, criterion, scheduler, config):
    best_iou = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.LOGS_DIR, "Train"), exist_ok=True)
    os.makedirs(os.path.join(config.LOGS_DIR, "Val"), exist_ok=True)

    train_writer = SummaryWriter(log_dir=os.path.join(config.LOGS_DIR, "Train"))
    val_writer = SummaryWriter(log_dir=os.path.join(config.LOGS_DIR, "Val"))

    trainer = TrainEpoch(model, criterion, optimizer, config.DEVICE, True, train_writer)
    validator = ValidEpoch(model, criterion, config.DEVICE, True, val_writer)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE_VAL, shuffle=False, num_workers=config.NUM_WORKERS)

    for epoch in range(1, config.MAX_EPOCHS + 1):
        print(f"[Epoch {epoch}/{config.MAX_EPOCHS}] LR: {scheduler.get_last_lr()[0]:.6f}")
        trainer.run(train_loader, i_epoch=epoch)

        if epoch % config.LOG_INTERVAL == 0:
            val_metrics = validator.run(val_loader, i_epoch=epoch)
            val_iou = round(val_metrics["iou_score"] * 100, 2)
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model, os.path.join(config.CHECKPOINT_DIR, f"model_best_epoch_{epoch:04d}.pth"))
                print(f"✅ Saved best model at epoch {epoch}")

        scheduler.step()

    train_writer.close()
    val_writer.close()
