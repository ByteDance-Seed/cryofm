from __future__ import annotations

import numpy as np
import mrcfile
import torch

__all__ = ["save_mrc"]


def save_mrc(file_path: str, data: torch.Tensor, voxel_size: float) -> None:
    """Save a 3D tensor to an MRC file.

    Args:
        file_path (str): Output file path.
        data (torch.Tensor): Volume data of shape ``(D, H, W)`` (CPU or CUDA).
        voxel_size (float): Voxel size written to the MRC header.

    Notes:
        Data is written as ``float32``.
    """
    data = data.detach()
    if data.is_cuda:
        data = data.cpu()

    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(data.numpy().astype(np.float32))
        mrc.voxel_size = voxel_size
        mrc.update_header_from_data()