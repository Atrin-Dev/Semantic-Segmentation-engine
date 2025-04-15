import os
import sys
import cv2
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp
import lookup_table as lut


class BaseSegmentationLoop:
    def __init__(self, model, loss_fn, device="cpu", verbose=True, writer=None, export_path=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.verbose = verbose
        self.writer = writer
        self.export_path = export_path

    def _log_format(self, log_dict):
        return ", ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])

    def _export_predictions(self, preds, target_paths, dataloader):
        if preds.device != dataloader.dataset.device:
            preds = preds.to(dataloader.dataset.device)
        for i, tgt_path in enumerate(target_paths):
            base_name = os.path.basename(tgt_path)
            fn_id = base_name.replace("_gtFine_labelIds.png", "_gtFine_predictionIds.png")
            fn_color = base_name.replace("_gtFine_labelIds.png", "_gtFine_color.png")

            pred_id = lut.lookup_chw(preds[i].byte(), dataloader.dataset.th_i_lut_trainid2id).permute(1, 2, 0).cpu().numpy()
            pred_color = lut.lookup_chw(preds[i].byte(), dataloader.dataset.th_i_lut_trainid2color).permute(1, 2, 0).cpu().numpy()
            pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(self.export_path, fn_id), pred_id)
            cv2.imwrite(os.path.join(self.export_path, fn_color), pred_color)

    def _write_tensorboard_logs(self, logs, images, targets, preds, dataloader, epoch_num):
        self.writer.add_scalar(f"Loss/{self.loss_fn.__name__}", logs[self.loss_fn.__name__], epoch_num)
        if "iou_score" in logs:
            self.writer.add_scalar("Metrics/IoU", logs["iou_score"], epoch_num)

        self.writer.add_images("Predicted/Color",
            lut.lookup_nchw(preds[:4].byte(), dataloader.dataset.th_i_lut_trainid2color), global_step=epoch_num)
        self.writer.add_images("GroundTruth/Color",
            lut.lookup_nchw(targets[:4].unsqueeze(1).byte(), dataloader.dataset.th_i_lut_trainid2color), global_step=epoch_num)
        self.writer.add_images("Input/Images",
            ((images[:4] + 2) * 64).round().clamp(0, 255).byte(), global_step=epoch_num)

        self.writer.flush()
