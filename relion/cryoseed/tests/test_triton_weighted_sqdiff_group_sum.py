from __future__ import annotations

import torch

from tests._triton_test_utils import CUDA_TRITON_REQUIRED

from cryoseed.backends.triton.primitives import weighted_sqdiff_group_sum_indexed_complex

pytestmark = CUDA_TRITON_REQUIRED


def _reference_group_sum(
    input: torch.Tensor,
    other: torch.Tensor,
    weight: torch.Tensor,
    input_indices: torch.Tensor,
    other_indices: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    proj_sel = input.index_select(0, input_indices)
    other_sel = other.index_select(0, other_indices)
    sqdiff = (proj_sel - other_sel).abs().square().to(torch.float32)
    weighted = sqdiff * weight.view(1, -1)
    out = torch.zeros((input_indices.numel(), num_groups), device=input.device, dtype=torch.float32)
    out.scatter_add_(
        1,
        group_index.view(1, -1).expand(input_indices.numel(), -1),
        weighted,
    )
    return out


def test_triton_weighted_sqdiff_group_sum_matches_reference():
    torch.manual_seed(11)
    device = torch.device("cuda")

    n_input, n_other, d, num_groups = 7, 6, 17, 5
    input = torch.randn(n_input, d, device=device, dtype=torch.complex64)
    other = torch.randn(n_other, d, device=device, dtype=torch.complex64)
    input_indices = torch.tensor([0, 2, 4, 6, 1], device=device, dtype=torch.int64)
    other_indices = torch.tensor([5, 3, 1, 0, 2], device=device, dtype=torch.int64)
    group_index = torch.tensor([0, 1, 1, 2, 3, 4, 0, 2, 3, 4, 1, 2, 0, 3, 4, 1, 2], device=device)
    group_scale = torch.tensor([1.0, 0.5, 0.25, 2.0, 1.5], device=device, dtype=torch.float32)
    weight = group_scale.index_select(0, group_index)

    out = weighted_sqdiff_group_sum_indexed_complex(
        input,
        other,
        weight,
        input_indices,
        other_indices,
        group_index,
        num_groups=num_groups,
    )
    ref = _reference_group_sum(
        input,
        other,
        weight,
        input_indices,
        other_indices,
        group_index,
        num_groups,
    )

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)