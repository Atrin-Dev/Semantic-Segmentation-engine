import torch
import torchvision
import numpy as np
import cv2
import lookup_table as lut


class DatasetCityscapesSemantic(torchvision.datasets.Cityscapes):
    def __init__(self, device, *args, **kwargs):
        """
        Custom Cityscapes dataset for semantic segmentation.

        Args:
            device (str): Device to use ('cuda' or 'cpu').
            *args, **kwargs: Passed to torchvision.datasets.Cityscapes.
        """
        # Force semantic target type
        super().__init__(*args, **kwargs, target_type="semantic")
        self.device = device

        # Build lookup tables for class ID ↔ train ID ↔ color
        ids, train_ids, colors = self._get_class_properties()

        self.id_to_trainid_lut = self._create_lookup(ids, train_ids, default=19)
        self.trainid_to_id_lut = self._create_lookup(train_ids, ids, default=0)
        self.trainid_to_color_lut = self._create_lookup(train_ids, colors, default=0)

    def _create_lookup(self, keys, values, default):
        """
        Helper to create lookup table tensors from key-value pairs.

        Args:
            keys (list of list of int): E.g., [[26], [27], ...]
            values (list of list or tuple): Same length as keys.
            default (int or list): Default value if key not found.

        Returns:
            torch.Tensor: Lookup table on the specified device.
        """
        np_keys = np.asarray(keys, dtype=np.uint8)
        np_values = np.asarray(values, dtype=np.uint8)
        _, lut_tensor = lut.get_lookup_table(
            ar_u_key=np_keys,
            ar_u_val=np_values,
            v_val_default=default,
            device=self.device,
        )
        return lut_tensor

    def _get_class_properties(self):
        """
        Collects relevant class metadata from Cityscapes class definitions.

        Returns:
            Tuple of three lists: IDs, train IDs, and RGB colors.
        """
        ids, train_ids, colors = [], [], []

        for class_info in self.classes:
            if class_info.train_id in [-1, 255]:
                continue
            ids.append([class_info.id])
            train_ids.append([class_info.train_id])
            colors.append(class_info.color)

        # Add background class
        ids.append([0])
        train_ids.append([19])
        colors.append([0, 0, 0])

        return ids, train_ids, colors

    def __getitem__(self, index):
        """
        Loads and returns a data sample with optional transforms.

        Returns:
            Tuple of (image, target_mask, image_path, target_path)
        """
        img_path = self.images[index]
        mask_path = self.targets[index][0]  # Only one target: semantic

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.transform:
            transformed = self.transform(image=image, mask=target)
            image = transformed["image"]
            target = transformed["mask"]

        return image, target, img_path, mask_path
