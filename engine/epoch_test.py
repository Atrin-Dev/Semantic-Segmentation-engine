from base import BaseSegmentationLoop
import torch
from tqdm import tqdm
import sys


class TestLoop(BaseSegmentationLoop):
    def execute(self, dataloader):
        self.model.eval()
        pbar = tqdm(dataloader, desc="test", file=sys.stdout, disable=not self.verbose)
        for images, _, _, tgt_paths in pbar:
            images = images.to(self.device)

            with torch.no_grad():
                preds = self.model(images).argmax(1, keepdim=True)

            if self.export_path:
                self._export_predictions(preds, tgt_paths, dataloader)
