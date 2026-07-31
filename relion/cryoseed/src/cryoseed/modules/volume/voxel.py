from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from cryoseed.fft.fft_torch import fourier_to_primal_3d
from cryoseed.utils.torch_utils import _norm_device

from cryoseed.config import MainConfig
from cryoseed.cryoem.mask import spherical_mask
from cryoseed.ops.transforms import (
    backproject as backproject_op,
    downsample3d,
    project as project_op,
)

from .volume import Volume

__all__ = [
    "VoxelGrid",
]


class VoxelGrid(Volume):
    """A learnable 3D Fourier voxel grid.

    The volume is stored as a complex tensor in Fourier space with shape
    ``(K, D, D, D)``, where:

    - ``D`` is ``grid_size`` (spatial side length).
    - ``K`` is ``num_volumes`` (number of volumes/classes).

    This module supports projecting the Fourier volume onto the z=0 central slice
    and accumulating backprojections into numerator/denominator buffers.
    """

    @classmethod
    def from_config(
        cls,
        config: MainConfig,
        device: torch.device | str | None = None,
        device_mesh: Any | None = None,
        *,
        requires_accum: bool = True,
        requires_grad: bool = False,
    ) -> VoxelGrid:
        return cls(
            grid_size=int(config.data.image_size),
            num_volumes=int(config.modules.volume.num_volumes),
            device=device,
            device_mesh=device_mesh,
            requires_accum=requires_accum,
            requires_grad=requires_grad,
            backproject_chunk=int(config.modules.volume.backproject_chunk),
        )

    def __init__(
        self,
        grid_size: int,
        num_volumes: int = 1,
        *,
        device: torch.device | str | None = None,
        device_mesh: Any | None = None,
        requires_accum: bool = True,
        requires_grad: bool = False,
        backproject_chunk: int = 65536,
    ):
        """Create a :class:`VoxelGrid`.

        Args:
            grid_size: Side length ``D`` of the cubic 3D grid.
            num_volumes: Number of volumes/classes ``K``.
            device: Device used to initialize parameters and buffers.
            device_mesh: Optional process group / device mesh used for all-reduce
                in :meth:`update` (parameter synchronization) and :meth:`backproject`
                (parallel backprojection accumulation).
            requires_accum: Whether to allocate accumulation buffers used by
                :meth:`backproject` and :meth:`update`.
            requires_grad: Whether ``self.volume`` requires gradients.
            backproject_chunk: Chunk size over the pose dimension used to reduce
                peak memory during backprojection.
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_volumes = num_volumes
        dev = _norm_device(device)
        self.register_buffer("_device_anchor", torch.empty(0, device=dev), persistent=False)
        self.device_mesh = device_mesh
        self.requires_accum = bool(requires_accum)
        self.requires_grad = bool(requires_grad)
        self.backproject_chunk = backproject_chunk

        self.volume = nn.Parameter(
            torch.zeros(
                (num_volumes, grid_size, grid_size, grid_size),
                device=dev,
                dtype=torch.complex64,
            ),
            requires_grad=requires_grad,
        )

        if requires_accum:
            accum_numer = torch.zeros_like(self.volume, dtype=torch.complex64)
            accum_denom = torch.zeros_like(self.volume, dtype=torch.float32)
        else:
            accum_numer = None
            accum_denom = None

        self.register_buffer(
            "accum_numer",
            accum_numer,
            persistent=False,
        )
        self.register_buffer(
            "accum_denom",
            accum_denom,
            persistent=False,
        )

        self.downsampled_volumes: dict[int, Tensor] = {}
        self.volume_version = 0

    @property
    def device(self) -> torch.device:
        """Device that backs module parameters and buffers."""
        return self._device_anchor.device

    def requires_grad_(self, requires_grad: bool = True) -> VoxelGrid:
        """Set autograd participation for the volume Parameter."""
        self.requires_grad = bool(requires_grad)
        self.volume.requires_grad_(self.requires_grad)
        return self

    @property
    def volume_real(self) -> Tensor:
        """Return the real part of the volume."""
        return fourier_to_primal_3d(self.volume).real

    def _sync_accum_tensor(self, tensor: Tensor) -> Tensor:
        out = tensor

        dist = getattr(torch, "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized():
            if self.device_mesh is not None:
                group = self.device_mesh.get_group(0) if hasattr(self.device_mesh, "get_group") else self.device_mesh
            else:
                group = dist.group.WORLD

            data_parallel_size = dist.get_world_size(group=group)
            if data_parallel_size > 1:
                out = out.clone()
                dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)

        return out

    @torch.no_grad()
    def sync_grad_(self) -> None:
        """Synchronize parameter gradients across the data-parallel group.

        Gradients are summed across ``device_mesh.get_group(0)`` when available
        (otherwise WORLD) and averaged in place so each replica steps with the
        same effective gradient.
        """
        dist = getattr(torch, "distributed", None)
        if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
            return

        if self.device_mesh is not None:
            group = (
                self.device_mesh.get_group(0)
                if hasattr(self.device_mesh, "get_group")
                else self.device_mesh
            )
        else:
            group = dist.group.WORLD

        data_parallel_size = dist.get_world_size(group=group)
        if data_parallel_size <= 1:
            return

        for param in self.parameters():
            grad = param.grad
            if grad is None:
                continue
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
            grad.div_(data_parallel_size)

    @property
    def accumulated_data(self) -> Tensor:
        """Return the accumulated per-voxel backprojection numerator.

        Returns:
            Complex tensor with shape ``(K, D, D, D)``.

        Note:
            If distributed training is initialized, the returned tensor is summed
            across the data-parallel synchronization group.
        """
        if (not self.requires_accum) or self.accum_numer is None:
            raise ValueError(
                "accumulated_data is not available for this VoxelGrid since "
                "requires_accum is False or accum_numer is None"
            )

        return self._sync_accum_tensor(self.accum_numer)

    @property
    def accumulated_weight(self) -> Tensor:
        """Return the accumulated per-voxel backprojection denominator.

        Returns:
            Real tensor with shape ``(K, D, D, D)``.

        Note:
            If distributed training is initialized, the returned tensor is summed
            across the data-parallel synchronization group.
        """
        if (not self.requires_accum) or self.accum_denom is None:
            raise ValueError(
                "accumulated_weight is not available for this VoxelGrid since "
                "requires_accum is False or accum_denom is None"
            )

        return self._sync_accum_tensor(self.accum_denom)

    @torch.no_grad()
    def load_volume(self, volume: Tensor) -> None:
        """Load a new Fourier volume tensor.

        Args:
            volume: Complex tensor with shape ``(K, D, D, D)`` matching
                ``(num_volumes, grid_size, grid_size, grid_size)``.
        """
        if volume.shape != self.volume.shape:
            raise ValueError(
                f"volume shape {tuple(volume.shape)} does not match "
                f"(num_volumes, grid_size, grid_size, grid_size)="
                f"{tuple(self.volume.shape)}"
            )

        volume = volume.detach()
        if volume.device != self.device:
            volume = volume.to(self.device)
        volume = volume.clone()

        self.volume = nn.Parameter(volume, requires_grad=self.requires_grad)

        if self.requires_accum:
            self.accum_numer = torch.zeros_like(self.volume, dtype=torch.complex64)
            self.accum_denom = torch.zeros_like(self.volume, dtype=torch.float32)
        else:
            self.accum_numer = None
            self.accum_denom = None

        self.downsampled_volumes.clear()
        self.volume_version += 1

    @torch.no_grad()
    def copy_volume_(self, volume: Tensor) -> None:
        """Copy a Fourier volume into the existing parameter in place.

        This keeps the current ``nn.Parameter`` object alive, so
        ``requires_grad``, any existing ``.grad`` tensor, and optimizer state
        attached to that parameter object are preserved. Accumulation buffers
        also remain intact, while cached downsampled views are invalidated.
        """
        if volume.shape != self.volume.shape:
            raise ValueError(
                f"volume shape {tuple(volume.shape)} does not match "
                f"reference shape {tuple(self.volume.shape)}"
            )
        self.volume.copy_(
            volume.detach().to(device=self.volume.device, dtype=self.volume.dtype)
        )
        self.downsampled_volumes.clear()
        self.volume_version += 1

    def _build_downsampled_volume(self, side_length: int) -> Tensor:
        downsampled_volume = downsample3d(self.volume, side_length)
        center = side_length // 2
        mask = spherical_mask(
            side_length,
            side_length,
            side_length,
            center=(center, center, center),
            radius=side_length / 2,
            device=downsampled_volume.device,
        )
        return downsampled_volume.masked_fill(~mask, 0.0)

    def get_downsampled_volumes(self, side_length: int) -> Tensor:
        """Return a cached Fourier volume cropped/downsampled to ``side_length``.

        The downsampled volume is masked with a spherical support up to the
        Nyquist radius.

        Args:
            side_length: Target side length ``L``.

        Returns:
            Complex tensor with shape ``(K, L, L, L)``.
        """
        if side_length not in self.downsampled_volumes:
            self.downsampled_volumes[side_length] = self._build_downsampled_volume(side_length)
        return self.downsampled_volumes[side_length]

    def _project(
        self,
        rotation: Tensor,
        side_length: int | None = None,
        *,
        use_cache: bool = True,
    ) -> Tensor:
        """Project the 3D Fourier volume onto the z=0 central slice.

        Args:
            rotation: Rotation matrices with shape ``(K, Q, 3, 3)``, where
                ``K == num_volumes`` and ``Q`` is the number of poses per volume.
                These rotations are defined for the volume relative to the
                detector (projection) frame.
            side_length: Output side length ``L``. If ``L < grid_size``, the stored
                Fourier volume is cropped/downsampled before projection.
            use_cache: Whether downsampled Fourier volumes may be reused from the
                internal cache. Differentiable callers should disable this so each
                invocation builds its own autograd graph.

        Returns:
            Complex tensor with shape ``(K, Q, L, L)``.

        Note:
            Backends operate on real tensors. Complex volumes are passed as
            ``(..., 2)`` with the last dimension storing ``(real, imag)``.
        """

        if side_length is None:
            side_length = self.grid_size

        if rotation.ndim != 4 or rotation.shape[-2:] != (3, 3):
            raise ValueError(f"rotation must be (K,Q,3,3), got {tuple(rotation.shape)}")

        if int(rotation.shape[0]) != int(self.num_volumes):
            raise ValueError(
                "rotation.shape[0] must equal num_volumes (K), "
                f"got {int(rotation.shape[0])} vs {int(self.num_volumes)}"
            )

        if side_length < self.grid_size:
            if use_cache:
                volume_cplx = self.get_downsampled_volumes(side_length)
            else:
                volume_cplx = self._build_downsampled_volume(side_length)
        else:
            volume_cplx = self.volume

        volume_ri = torch.view_as_real(volume_cplx).contiguous()

        proj_ri = project_op(volume_ri, rotation, channel_last=True)
        return torch.view_as_complex(proj_ri.contiguous())

    @torch.no_grad()
    def project(
        self,
        rotation: Tensor,
        side_length: int | None = None,
    ) -> Tensor:
        """Project Fourier volumes to the z=0 central slice without autograd.

        This is the default inference/search entry point. It shares the same
        implementation as :meth:`forward` by calling :meth:`_project`, but runs
        under ``torch.no_grad()``.

        Args:
            rotation: Rotation matrices with shape ``(K, Q, 3, 3)``, where
                ``K == num_volumes`` and ``Q`` is the number of poses per volume.
            side_length: Output side length ``L``. If ``L < grid_size``, the stored
                Fourier volumes are downsampled/cropped to match the requested
                resolution.

        Returns:
            Complex tensor with shape ``(K, Q, L, L)``.
        """
        return self._project(
            rotation,
            side_length=side_length,
            use_cache=True,
        )
    
    @torch.no_grad()
    def backproject(
        self,
        image: Tensor,
        ctf: Tensor | None,
        probability: Tensor | None,
        image_index: Tensor | None,
        volume_index: Tensor | None,
        rotation: Tensor,
        translation: Tensor,
        *,
        noise_spectrum: Tensor | None = None,
        radius: float | None = None,
    ) -> None:
        """Accumulate backprojections into internal numerator/denominator buffers.

        This method is intended to be called repeatedly to accumulate evidence
        from many images/poses. It updates ``self.accum_numer`` and
        ``self.accum_denom`` in-place.

        Parallel backprojection:
            If ``torch.distributed`` is initialized and ``device_mesh`` is
            provided, this method shards the pose dimension ``N`` across a
            *calculation* process group (``device_mesh.get_group(1)``).

            Each rank backprojects only its slice of poses into per-call local
            buffers (``local_numer/local_denom``). These local buffers are then
            summed across the calculation group via ``all_reduce`` and finally
            added into the persistent ``accum_numer/accum_denom`` buffers.

            This design avoids all-reducing the pre-existing accumulation buffers
            (which may already contain contributions from previous calls).

        How this differs from :meth:`update`:
            - The ``all_reduce`` inside :meth:`backproject` combines *new per-call
              work* across ranks and updates the accumulation buffers.
            - The ``all_reduce`` inside :meth:`update` synchronizes the current
              accumulators across data-parallel replicas (typically
              ``device_mesh.get_group(0)``) before computing ``volume = numer / denom``.
              It reduces into temporary clones and does not modify the internal buffers.

        Args:
            image: Input Fourier images with shape ``(B, D, D)`` and dtype
                ``torch.complex64``.
            ctf: Optional per-image CTF modulation with shape ``(B, D, D)``.
            probability: Optional per-pose weights with shape ``(N,)``.
            image_index: Optional pose-to-image index mapping with shape ``(N,)``.
                If ``None``, requires ``N == B``.
            volume_index: Optional pose-to-volume index mapping with shape ``(N,)``.
                If ``None``, all poses are accumulated into volume 0.
            rotation: Per-pose rotation matrices with shape ``(N, 3, 3)`` or
                flattened shape ``(N, 9)``.
            translation: Per-pose translations with shape ``(N, 2)``.
            noise_spectrum: Optional per-pixel noise spectrum with shape ``(D, D)``.
                Defaults to ones.
            radius: Maximum radial support in Fourier pixels. Defaults to ``D // 2``.

        Raises:
            RuntimeError: If ``requires_accum=False``.
            ValueError: If ``image`` has incompatible dtype/shape.
        """
        if not self.requires_accum:
            raise RuntimeError("backproject requires requires_accum=True")

        if image.dtype != torch.complex64:
            raise ValueError(f"image must be complex64, got {image.dtype}")
        if image.dim() != 3 or image.shape[1] != image.shape[2]:
            raise ValueError(f"image must be (B,D,D), got {tuple(image.shape)}")

        _, D, _ = image.shape
        if D != self.grid_size:
            raise ValueError(f"image shape {D} does not match grid_size {self.grid_size}")
        
        if radius is None:
            radius = float(D // 2)

        if noise_spectrum is None:
            noise_spectrum = torch.ones((D, D), device=image.device, dtype=torch.float32)

        if self.accum_numer is None or self.accum_numer.shape != self.volume.shape:
            self.accum_numer = torch.zeros_like(self.volume, dtype=torch.complex64)
        if self.accum_denom is None or self.accum_denom.shape != self.volume.shape:
            self.accum_denom = torch.zeros_like(self.volume, dtype=torch.float32)

        local_numer = torch.zeros_like(self.accum_numer)
        local_denom = torch.zeros_like(self.accum_denom)

        dist = getattr(torch, "distributed", None)

        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and self.device_mesh is not None
        ):
            group = self.device_mesh.get_group(1)

            calculation_parallel_size = dist.get_world_size(group)
            calculation_process_rank = dist.get_rank(group)
        else:
            group = None
            calculation_parallel_size = 1
            calculation_process_rank = 0

        slice_size = math.ceil(rotation.shape[0] / calculation_parallel_size)

        slice_start = slice_size * calculation_process_rank
        slice_end = min(slice_size * (calculation_process_rank + 1), rotation.shape[0])
        
        for chunk_start in range(slice_start, slice_end, self.backproject_chunk):
            chunk_end = min(chunk_start + self.backproject_chunk, slice_end)
            prob = probability[chunk_start:chunk_end] if probability is not None else None
            img_idx = image_index[chunk_start:chunk_end] if image_index is not None else None
            vol_idx = volume_index[chunk_start:chunk_end] if volume_index is not None else None
            rot = rotation[chunk_start:chunk_end]
            trans = translation[chunk_start:chunk_end]

            backproject_op(
                image=image,
                ctf=ctf,
                noise_spectrum=noise_spectrum,
                probability=prob,
                image_index=img_idx,
                volume_index=vol_idx,
                rotation=rot,
                translation=trans,
                radius=radius,
                volume_numerator=local_numer,
                volume_denominator=local_denom,
                return_denom=True
            )

        if (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and calculation_parallel_size > 1
        ):
            dist.all_reduce(local_numer, dist.ReduceOp.SUM, group=group)
            dist.all_reduce(local_denom, dist.ReduceOp.SUM, group=group)

        self.accum_numer += local_numer
        self.accum_denom += local_denom


    @torch.no_grad()
    def update(self, prior_precision_spectrum: Tensor | None = None) -> None:
        """Update ``self.volume`` from accumulated backprojections.

        This computes ``volume = accum_numer / (accum_denom + prior)`` (where valid),
        clears cached downsampled volumes, and increments ``volume_version``.

        Args:
            prior_precision_spectrum: Optional real-valued prior precision spectrum
                broadcastable to ``(K, D, D, D)``. Common shapes are ``(D, D, D)``,
                ``(1, D, D, D)``, or ``(K, D, D, D)``.

        Notes:
            If distributed training is initialized and multiple ranks are present,
            the accumulators are all-reduced across the parameter synchronization
            group (``device_mesh.get_group(0)`` when available, otherwise WORLD).
            The reduction is performed into temporary clones so in-module buffers
            are left unchanged.

        """
        if (not self.requires_accum) or self.accum_numer is None or self.accum_denom is None:
            return

        numer = self.accum_numer
        denom = self.accum_denom

        dist = getattr(torch, "distributed", None)
        if dist is not None and dist.is_available() and dist.is_initialized():
            if self.device_mesh is not None:
                group = self.device_mesh.get_group(0) if hasattr(self.device_mesh, "get_group") else self.device_mesh
            else:
                group = dist.group.WORLD

            data_parallel_size = dist.get_world_size(group=group)
            if data_parallel_size > 1:
                numer = numer.clone()
                denom = denom.clone()
                dist.all_reduce(numer, op=dist.ReduceOp.SUM, group=group)
                dist.all_reduce(denom, op=dist.ReduceOp.SUM, group=group)

        denom_eff = denom
        if prior_precision_spectrum is not None:
            prior = prior_precision_spectrum.to(device=denom.device, dtype=denom.dtype)
            if prior.ndim == 3:
                prior = prior.unsqueeze(0)

            if prior.shape == denom.shape:
                pass
            elif prior.ndim == 4 and prior.shape[0] == 1 and tuple(prior.shape[1:]) == tuple(denom.shape[1:]):
                prior = prior.expand_as(denom)
            else:
                raise ValueError(
                    "prior_precision_spectrum must be broadcastable to "
                    f"{tuple(denom.shape)}, got {tuple(prior.shape)}"
                )

            denom_eff = denom + prior

        valid = denom_eff > 1e-9
        torch.div(numer, denom_eff, out=self.volume)
        self.volume.masked_fill_(~valid, 0)
        self.downsampled_volumes.clear()
        self.volume_version += 1
    
    @torch.no_grad()
    def zero_accum(self, *, set_to_none: bool = False) -> None:
        """Reset accumulation buffers.

        Args:
            set_to_none: If ``True``, set accumulator buffers to ``None``.
                Otherwise, when ``requires_accum=True``, allocate (if needed)
                and zero them.
        """
        if set_to_none:
            self.accum_numer = None
            self.accum_denom = None
            return

        if not self.requires_accum:
            return

        if self.accum_numer is None or self.accum_numer.shape != self.volume.shape:
            self.accum_numer = torch.zeros_like(self.volume, dtype=torch.complex64)
        if self.accum_denom is None or self.accum_denom.shape != self.volume.shape:
            self.accum_denom = torch.zeros_like(self.volume, dtype=torch.float32)

        self.accum_numer.zero_()
        self.accum_denom.zero_()

    def forward(self, rotation: Tensor, side_length: int | None = None) -> Tensor:
        """Differentiable projection path for module-style invocation.

        Unlike :meth:`project`, this method does not run under ``torch.no_grad()``.
        It calls the shared :meth:`_project` implementation with
        ``use_cache=False`` so gradients can flow back to ``self.volume`` without
        reusing stale cached downsampled volumes.

        Args:
            rotation: Rotation matrices with shape ``(K, Q, 3, 3)``.
            side_length: Output side length ``L``. If omitted, uses
                ``grid_size``.

        Returns:
            Complex tensor with shape ``(K, Q, L, L)``.
        """
        return self._project(rotation, side_length=side_length, use_cache=False)