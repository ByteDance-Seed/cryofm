from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["DataBatch"]


@dataclass(slots=True)
class DataBatch:
    """A framework-level batch consumed by solvers.

    This is the standardized batch produced by the data pipeline and passed into
    ``solver.infer(batch)``.

    Attributes:
        image (torch.Tensor):
            Particle images in Fourier space (fftshift convention), shape ``(B, D, D)``.
            Typically ``complex64``.
        image_real (torch.Tensor):
            Particle images in real space, shape ``(B, D, D)``.
        particle_index (torch.Tensor):
            Dataset-level particle indices, shape ``(B,)``.
        ctf_params (torch.Tensor | None):
            Per-particle CTF parameters, shape ``(B, C)``, or ``None``.
            The convention of the last dimension is defined by the data pipeline.
        ctf (torch.Tensor | None):
            Optional precomputed CTF in Fourier space, shape ``(B, D, D)``.
        stack_index (torch.Tensor | None):
            Optional per-particle index within the source MRC/MRCS stack, shape ``(B,)``.
        euler (torch.Tensor | None):
            Optional Euler angles in radians, shape ``(B, 3)``.
        trans (torch.Tensor | None):
            Optional in-plane translations in pixels, shape ``(B, 2)``.
    """

    image: torch.Tensor
    image_real: torch.Tensor
    particle_index: torch.Tensor
    ctf_params: torch.Tensor | None = None
    ctf: torch.Tensor | None = None

    stack_index: torch.Tensor | None = None
    euler: torch.Tensor | None = None
    trans: torch.Tensor | None = None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "DataBatch":
        """Return a new batch moved to the given device.

        Args:
            device (torch.device | str): Target device.
            non_blocking (bool, optional): Whether to perform the copy asynchronously
                when possible. Defaults to ``False``.

        Returns:
            DataBatch: A new batch with all tensor fields moved to ``device``.
        """
        return DataBatch(
            image=self.image.to(device, non_blocking=non_blocking),
            image_real=self.image_real.to(device, non_blocking=non_blocking),
            particle_index=self.particle_index.to(device, non_blocking=non_blocking),
            ctf_params=None
            if self.ctf_params is None
            else self.ctf_params.to(device, non_blocking=non_blocking),
            ctf=None
            if self.ctf is None
            else self.ctf.to(device, non_blocking=non_blocking),
            stack_index=None
            if self.stack_index is None
            else self.stack_index.to(device, non_blocking=non_blocking),
            euler=None
            if self.euler is None
            else self.euler.to(device, non_blocking=non_blocking),
            trans=None
            if self.trans is None
            else self.trans.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "DataBatch":
        """Return a new batch with all tensor fields pinned in CPU memory.

        Returns:
            DataBatch: A new batch with pinned tensor fields.
        """
        return DataBatch(
            image=self.image.pin_memory(),
            image_real=self.image_real.pin_memory(),
            particle_index=self.particle_index.pin_memory(),
            ctf_params=None if self.ctf_params is None else self.ctf_params.pin_memory(),
            ctf=None if self.ctf is None else self.ctf.pin_memory(),
            stack_index=None if self.stack_index is None else self.stack_index.pin_memory(),
            euler=None if self.euler is None else self.euler.pin_memory(),
            trans=None if self.trans is None else self.trans.pin_memory(),
        )

    @property
    def batch_size(self) -> int:
        return int(self.image.shape[0])