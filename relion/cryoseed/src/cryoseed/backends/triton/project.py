from __future__ import annotations

from torch import Tensor

from cryoseed.backends.triton.primitives import central_slice_sample


__all__ = [
    "project",
]


def project(volume: Tensor, rotation: Tensor, channel_last: bool = True) -> Tensor:
    """Project a 3D Fourier volume by sampling its z=0 central slice (Triton).

    Args:
        volume:
            Input volume tensor.

            - ``channel_last=True``: ``(B, D, H, W, C)`` (NDHWC)
            - ``channel_last=False``: ``(B, C, D, H, W)`` (NCDHW)

            Typically ``C=2`` stores complex numbers as ``(re, im)``.
        rotation:
            Rotation matrices with shape ``(B, Q, 3, 3)``.

            The rotation represents the rotation of the *volume* relative to the
            detector frame. Coordinates are treated as row-vectors and
            right-multiplied (``coords_rot = coords @ R``).
        channel_last:
            Whether the channel dimension is last (NDHWC) or second (NCDHW).

    Returns:
        Projected central slices.

        - ``channel_last=True``: ``(B, Q, H, W, C)``
        - ``channel_last=False``: ``(B, Q, C, H, W)``

    Notes:
        This uses ``align_dc=True`` internally to match the centered FFT frequency
        convention (for even ``L`` the DC frequency lies between pixels).
    """

    if not volume.is_cuda:
        raise RuntimeError("Triton project requires CUDA tensors")

    if rotation.device != volume.device:
        rotation = rotation.to(device=volume.device)

    return central_slice_sample(volume, rotation, channel_last=channel_last, align_dc=True)