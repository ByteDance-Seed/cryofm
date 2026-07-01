from __future__ import annotations

import pytest
import torch

from tests._triton_test_utils import CUDA_TRITON_REQUIRED, random_rotation

from cryoseed.backends.torch.backproject import backproject as backproject_torch
from cryoseed.backends.triton.backproject import backproject as backproject_triton

pytestmark = CUDA_TRITON_REQUIRED


@pytest.mark.parametrize("indexed", [False, True])
def test_triton_backproject_preserves_partial_sums_across_autotune_keys(indexed: bool):
    torch.manual_seed(17)
    device = torch.device("cuda")

    l = 18
    b = 3
    n = 5 if indexed else b

    image = torch.randn(b, l, l, device=device, dtype=torch.complex64)
    ctf = torch.randn(b, l, l, device=device, dtype=torch.float32)
    noise_spectrum = 0.25 + torch.rand(l, l, device=device, dtype=torch.float32)
    rotation = random_rotation(1, n, device=device).reshape(n, 3, 3)
    translation = 0.15 * torch.randn(n, 2, device=device, dtype=torch.float32)
    probability = 0.2 + torch.rand(n, device=device, dtype=torch.float32)

    image_index = None
    if indexed:
        image_index = torch.tensor([0, 2, 1, 2, 0], device=device, dtype=torch.int64)

    vol_num_triton = torch.zeros((1, l, l, l), device=device, dtype=torch.complex64)
    vol_den_triton = torch.zeros((1, l, l, l), device=device, dtype=torch.float32)
    vol_num_torch = torch.zeros_like(vol_num_triton)
    vol_den_torch = torch.zeros_like(vol_den_triton)

    # Use two different radii so the second launch sees a new (L, P) autotune key
    # while reusing the same accumulation buffer populated by the first launch.
    for radius in (5.0, 7.0):
        backproject_triton(
            image=image,
            ctf=ctf,
            noise_spectrum=noise_spectrum,
            image_index=image_index,
            volume_index=None,
            rotation=rotation,
            translation=translation,
            probability=probability,
            radius=radius,
            volume_numerator=vol_num_triton,
            volume_denominator=vol_den_triton,
            return_denom=True,
        )
        backproject_torch(
            image=image,
            ctf=ctf,
            noise_spectrum=noise_spectrum,
            image_index=image_index,
            volume_index=None,
            rotation=rotation,
            translation=translation,
            probability=probability,
            radius=radius,
            volume_numerator=vol_num_torch,
            volume_denominator=vol_den_torch,
            return_denom=True,
        )

    torch.testing.assert_close(vol_num_triton, vol_num_torch, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(vol_den_triton, vol_den_torch, rtol=3e-3, atol=3e-3)