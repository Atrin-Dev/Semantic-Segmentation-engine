from base import BaseSegmentationLoop
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import sys


class TrainLoop(BaseSegmentationLoop):
    def __init__(self, model, loss_fn, optimizer, device="cpu", verbose=True, writer=None):
        super().__init__(model, loss_fn, device, verbose, writer)
        self.optimizer = optimizer

    def execute(self, dataloader, epoch_num=-1):
        self.model.train()
        logs = {}
        total_loss = 0.0
        iteration_count = 0
        confusion_stats = {"tp": None, "fp": None, "fn": None, "tn": None}

        pbar = tqdm(dataloader, desc="train", file=sys.stdout, disable=not self.verbose)
        for images, targets, _, _ in pbar:
            images = images.to(self.device)
            targets = targets.unsqueeze(1).to(dataloader.dataset.device)
            targets = lut.lookup_nchw(targets, dataloader.dataset.th_i_lut_id2trainid).squeeze(1).long().to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.loss_fn(preds, targets)
            loss.backward()
            self.optimizer.step()

            iteration_count += 1
            total_loss += loss.item()
            logs[self.loss_fn.__name__] = total_loss / iteration_count
            pbar.set_postfix_str(self._log_format(logs))

            preds = preds.argmax(1, keepdim=True)
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds.squeeze(1),
                targets,
                mode="multiclass",
                num_classes=19,
                ignore_index=19
            )
            for k, stat in zip(["tp", "fp", "fn", "tn"], [tp, fp, fn, tn]):
                if confusion_stats[k] is None:
                    confusion_stats[k] = stat.sum(0, keepdim=True)
                else:
                    confusion_stats[k] += stat.sum(0, keepdim=True)

        logs["iou_score"] = smp.metrics.functional.iou_score(
            tp=confusion_stats["tp"],
            fp=confusion_stats["fp"],
            fn=confusion_stats["fn"],
            tn=confusion_stats["tn"],
            reduction="macro-imagewise"
        ).cpu().numpy()

        if self.writer:
            self._write_tensorboard_logs(logs, images, targets, preds, dataloader, epoch_num)

        return logs
