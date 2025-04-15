import torch
import numpy as np


def build_lookup_table(keys_np: np.ndarray, values_np: np.ndarray, device, default_value=0):
    """
    Construct a lookup table from key-value pairs, where keys are multi-channel uint8 (e.g. RGB)
    and values are uint8 label vectors.

    Args:
        keys_np (np.ndarray): shape (num_keys, num_channels), dtype=uint8
        values_np (np.ndarray): shape (num_keys, value_dim), dtype=uint8
        device (torch.device): where to move the PyTorch tensor
        default_value (int): default value for unknown keys

    Returns:
        lut_np (np.ndarray): full lookup table as numpy array
        lut_tensor (torch.Tensor): lookup table as torch tensor on given device
    """
    if keys_np.shape[0] != values_np.shape[0]:
        raise ValueError("Number of keys and values must match.")
    if keys_np.dtype != np.uint8 or values_np.dtype != np.uint8:
        raise ValueError("Both keys and values must be uint8.")

    num_channels = keys_np.shape[1]
    output_dim = values_np.shape[1]

    powers = (256 ** np.arange(num_channels)).astype(np.int32)  # shape: (num_channels,)
    keys_encoded = (keys_np.astype(np.int32) * powers).sum(axis=1)  # shape: (num_keys,)

    lut_size = 256 ** num_channels
    lut_np = np.full((lut_size, output_dim), default_value, dtype=np.int32)
    lut_np[keys_encoded] = values_np

    lut_tensor = torch.from_numpy(lut_np).to(device)

    return lut_np, lut_tensor


def lookup_chw(img_tensor: torch.Tensor, lut_tensor: torch.Tensor):
    """
    Apply LUT to a single image tensor with shape (C, H, W)

    Args:
        img_tensor (torch.Tensor): input tensor, dtype=uint8, shape (C, H, W)
        lut_tensor (torch.Tensor): lookup table, dtype=int32, shape (lut_size, value_dim)

    Returns:
        torch.Tensor: transformed tensor, dtype=uint8, shape (value_dim, H, W)
    """
    if img_tensor.dtype != torch.uint8 or lut_tensor.dtype != torch.int32:
        raise ValueError("img_tensor must be uint8 and lut_tensor must be int32")
    if img_tensor.ndim != 3:
        raise ValueError("img_tensor must be 3D (C, H, W)")
    if img_tensor.device != lut_tensor.device:
        raise ValueError("img_tensor and lut_tensor must be on the same device")

    num_channels = img_tensor.shape[0]
    lut_dim = lut_tensor.shape[1]

    # Compute encoded index from channels
    weights = (256 ** torch.arange(num_channels, device=img_tensor.device)).view(-1, 1, 1)
    encoded = (img_tensor.to(torch.int32) * weights).sum(dim=0)  # shape: (H, W)

    # Apply LUT per output channel
    out_tensor = torch.zeros((lut_dim, *encoded.shape), dtype=torch.uint8, device=img_tensor.device)
    for c in range(lut_dim):
        out_tensor[c] = torch.take(lut_tensor[:, c], encoded)

    return out_tensor


def lookup_nchw(batch_tensor: torch.Tensor, lut_tensor: torch.Tensor):
    """
    Apply LUT to a batch of image tensors with shape (N, C, H, W)

    Args:
        batch_tensor (torch.Tensor): input tensor, dtype=uint8, shape (N, C, H, W)
        lut_tensor (torch.Tensor): lookup table, dtype=int32, shape (lut_size, value_dim)

    Returns:
        torch.Tensor: transformed tensor, dtype=uint8, shape (N, value_dim, H, W)
    """
    if batch_tensor.dtype != torch.uint8 or lut_tensor.dtype != torch.int32:
        raise ValueError("batch_tensor must be uint8 and lut_tensor must be int32")
    if batch_tensor.ndim != 4:
        raise ValueError("batch_tensor must be 4D (N, C, H, W)")
    if batch_tensor.device != lut_tensor.device:
        raise ValueError("batch_tensor and lut_tensor must be on the same device")

    N, C, H, W = batch_tensor.shape
    lut_dim = lut_tensor.shape[1]

    weights = (256 ** torch.arange(C, device=batch_tensor.device)).view(1, C, 1, 1)
    encoded = (batch_tensor.to(torch.int32) * weights).sum(dim=1)  # shape: (N, H, W)

    out_tensor = torch.zeros((N, lut_dim, H, W), dtype=torch.uint8, device=batch_tensor.device)
    for c in range(lut_dim):
        out_tensor[:, c] = torch.take(lut_tensor[:, c], encoded)

    return out_tensor
