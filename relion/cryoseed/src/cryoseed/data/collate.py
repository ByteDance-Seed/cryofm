from __future__ import annotations

import torch

from cryoseed.cryoem.ctf import ctf_from_params
from cryoseed.fft.fft_torch import primal_to_fourier_2d

from .batch import DataBatch

__all__ = ["data_collate_fn"]


def data_collate_fn(
    samples: list[dict[str, torch.Tensor | int | float | None]],
) -> DataBatch:
    """Collate per-sample dicts into a :class:`~cryoseed.data.DataBatch`.

    Args:
        samples: A list of sample dicts produced by :class:`~cryoseed.data.ParticleDataset`.
            Required keys:

            - ``"image_real"`` (torch.Tensor): Real-space image, shape ``(D, D)``.
            - ``"particle_index"`` (int): Dataset-level particle index.

            Optional keys (must be either present for all samples in the batch or absent for all):

            - ``"stack_index"`` (int): Index within the source MRC/MRCS stack.
            - ``"euler"`` (torch.Tensor | None): Euler angles in radians, shape ``(3,)``.
            - ``"trans"`` (torch.Tensor | None): In-plane translation in pixels, shape ``(2,)``.
            - ``"ctf_params"`` (torch.Tensor | None): CTF parameter vector, shape ``(C,)``.

            If ``ctf_params`` is provided, ``"angpix"`` (float, Å/pixel) must be present and
            consistent across the batch to compute a per-image CTF.

    Returns:
        DataBatch: A batch where tensor fields are stacked along the batch dimension.
            Optional fields are returned as ``None`` when absent.

    Raises:
        ValueError: If ``samples`` is empty, if optional field presence is inconsistent
            across the batch, or if ``angpix`` is missing/inconsistent when computing CTF.
    """
    if len(samples) == 0:
        raise ValueError("Cannot collate an empty batch.")

    image_real = torch.stack([sample["image_real"] for sample in samples], dim=0)
    image = primal_to_fourier_2d(image_real)

    particle_index = torch.tensor(
        [sample["particle_index"] for sample in samples],
        dtype=torch.long,
    )

    stack_index = torch.tensor(
        [sample["stack_index"] for sample in samples],
        dtype=torch.long,
    )

    euler_list: list[torch.Tensor | None] = [sample.get("euler", None) for sample in samples]
    if all(t is None for t in euler_list):
        euler = None
    elif any(t is None for t in euler_list):
        raise ValueError("Inconsistent batch: some samples have euler, others do not.")
    else:
        euler = torch.stack([t for t in euler_list if t is not None], dim=0)

    trans_list: list[torch.Tensor | None] = [sample.get("trans", None) for sample in samples]
    if all(t is None for t in trans_list):
        trans = None
    elif any(t is None for t in trans_list):
        raise ValueError("Inconsistent batch: some samples have trans, others do not.")
    else:
        trans = torch.stack([t for t in trans_list if t is not None], dim=0)

    ctf_params_list: list[torch.Tensor | None] = [
        sample.get("ctf_params", None) for sample in samples
    ]
    if all(ctf_params is None for ctf_params in ctf_params_list):
        ctf_params = None
        ctf = None
    elif any(ctf_params is None for ctf_params in ctf_params_list):
        raise ValueError(
            "Inconsistent batch: some samples have ctf_params, others do not."
        )
    else:
        ctf_params = torch.stack([t for t in ctf_params_list if t is not None], dim=0)

        angpix = samples[0].get("angpix", None)
        if angpix is None:
            raise ValueError(
                "Cannot compute CTF in collate: samples are missing 'angpix'."
            )
        for sample in samples[1:]:
            angpix_i = sample.get("angpix", None)
            if angpix_i is None:
                raise ValueError(
                    "Cannot compute CTF in collate: samples are missing 'angpix'."
                )
            if float(angpix_i) != float(angpix):
                raise ValueError(
                    f"Inconsistent angpix in batch: {float(angpix)} vs {float(angpix_i)}"
                )

        side_length = int(image.shape[-1])
        ctf = ctf_from_params(
            ctf_params,
            side_length=side_length,
            angpix=float(angpix),
            device=image.device,
        )

    return DataBatch(
        image=image,
        image_real=image_real,
        particle_index=particle_index,
        stack_index=stack_index,
        euler=euler,
        trans=trans,
        ctf_params=ctf_params,
        ctf=ctf,
    )