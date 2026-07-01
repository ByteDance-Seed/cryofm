import logging
import math
import os
import warnings
from typing import Any

import numpy as np
import torch

from cryoseed.utils.torch_utils import _norm_device

from cryoseed.config import MainConfig
from cryoseed.cryoem.mask import circular_mask
from cryoseed.cryoem.rotation import quaternion_to_matrix
from cryoseed.fft.coords import fftindex_radial2d
from cryoseed.modules.pose import Pose
from cryoseed.modules.statistics.noise import NoiseVariance
from cryoseed.modules.volume import Volume
from cryoseed.ops.loss import spectral_mse_loss
from cryoseed.ops.transforms import downsample2d, translate_image
from cryoseed.state import OptimState

from . import shift_grid, so3_grid

from .cache import MemoryProjCache, SSDProjCache


LOGGER = logging.getLogger(__name__)


def pack_proj_cache_key(uid: int, healpix_order: int, side_length: int) -> int:
    uid = int(uid)
    healpix_order = int(healpix_order)
    side_length = int(side_length)

    if uid < 0:
        raise ValueError("uid must be >= 0")
    if not (0 <= healpix_order < 16):
        raise ValueError("healpix_order must be in [0, 16)")
    if not (0 <= side_length < 1024):
        raise ValueError("side_length must be in [0, 1024)")

    return (uid << 14) | (side_length << 4) | healpix_order


def numpy_to_tensor(
    x: np.ndarray | torch.Tensor | None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        raise TypeError(f"Expected numpy array or torch tensor, got {type(x)}")

    need_dtype = dtype is not None and t.dtype != dtype
    need_device = device is not None and t.device != device
    if need_dtype or need_device:
        t = t.to(
            device=device if device is not None else t.device,
            dtype=dtype if dtype is not None else t.dtype,
        )
    return t


class HEALPixPoseSearcher(torch.nn.Module):

    @classmethod
    def from_config(
        cls,
        state: OptimState,
        volume: Volume,
        pose: Pose,
        *,
        config: MainConfig,
        noise: NoiseVariance | None = None,
        device: torch.device | str | None = None,
        device_mesh=None,
    ):
        return cls(
            state=state,
            volume=volume,
            pose=pose,
            noise=noise,
            device=device,
            device_mesh=device_mesh,
            trans_grid_samples=config.pose_search.trans_grid_samples,
            trans_grid_x_shift=config.pose_search.trans_grid_x_shift,
            trans_grid_y_shift=config.pose_search.trans_grid_y_shift,
            pose_chunk_factor=config.pose_search.pose_chunk_factor,
            max_candidates=config.pose_search.max_candidates,
            mse_chunk=config.pose_search.mse_chunk,
            candidate_select_threshold=config.pose_search.candidate_select_threshold,
            renormalize_sel_prob=config.pose_search.renormalize_sel_prob,
            oversampling_deduplicate=config.pose_search.oversampling_deduplicate,
            ring_averaged_mse=config.pose_search.ring_averaged_mse,
            ssd_cache_root=config.io.ssd_cache_root,
        )

    def __init__(
        self,
        state: OptimState,
        volume: Volume,
        pose: Pose,
        *,
        noise: NoiseVariance | None = None,
        device: torch.device | str | None = None,
        device_mesh: Any | None = None,
        trans_grid_samples: int = 5,
        trans_grid_x_shift: int = 0,
        trans_grid_y_shift: int = 0,
        pose_chunk_factor: int = 2560,
        max_candidates: int = -1,
        mse_chunk: int = 8192,
        candidate_select_threshold: float = 0.999,
        renormalize_sel_prob: bool = True,
        oversampling_deduplicate: bool = False,
        ring_averaged_mse: bool = False,
        ssd_cache_root: str | None = None,
    ):
        super().__init__()

        self.state = state
        self.volume = volume
        self.pose = pose
        self.noise = noise
        dev = _norm_device(device)
        self.register_buffer("_device_anchor", torch.empty(0, device=dev), persistent=False)
        self.device_mesh = device_mesh

        # fixed parameters
        self.trans_grid_samples = trans_grid_samples
        self.trans_grid_x_shift = trans_grid_x_shift
        self.trans_grid_y_shift = trans_grid_y_shift
        self.pose_chunk_factor = pose_chunk_factor
        self.mse_chunk = mse_chunk
        self.candidate_select_threshold = candidate_select_threshold
        self.renormalize_sel_prob = renormalize_sel_prob
        self.oversampling_deduplicate = bool(oversampling_deduplicate)
        self.ring_averaged_mse = bool(ring_averaged_mse)
        self.ssd_cache_root = ssd_cache_root

        if self.pose_chunk_factor is not None and self.pose_chunk_factor <= 0:
            raise ValueError("pose_chunk_factor must be positive or None")
        if self.mse_chunk <= 0:
            raise ValueError("mse_chunk must be > 0")
        if max_candidates == 0 or max_candidates < -1:
            raise ValueError("max_candidates must be > 0, or -1 for unlimited")
        self.max_candidates = None if max_candidates == -1 else int(max_candidates)
        if not (0 < self.candidate_select_threshold <= 1):
            raise ValueError("candidate_select_threshold must be in (0, 1]")


        # placeholder
        self.memory_cache = None
        self.ssd_cache = None
        self.volume_version = None
        # dynamic parameters
        self.refresh()

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device


    def _set_buffer(
        self,
        name: str,
        value: torch.Tensor,
        *,
        persistent: bool = False,
    ) -> None:
        """Register or overwrite a module buffer.

        Args:
            name: Buffer name.
            value: Tensor value.
            persistent: Whether the buffer should be included in the state dict.
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Buffer '{name}' must be a torch.Tensor, got {type(value)}")

        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value, persistent=persistent)

        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def _refresh_schedule_state(self) -> None:
        schedule = self.state.schedule

        self.side_length = schedule.side_length
        self.base_healpix_order = schedule.healpix_order
        self.current_healpix_order = schedule.healpix_order
        self.base_trans_healpix_order = 0
        self.current_trans_healpix_order = 0
        self.num_oversampling = schedule.oversampling
        self.trans_grid_extent = float(schedule.trans_grid_extent)

        if int(self.base_healpix_order) < 1:
            raise ValueError("healpix_order must be >= 1")
        if self.trans_grid_extent < 0:
            raise ValueError("trans_grid_extent must be >= 0")

    def _refresh_pose_grid(self) -> None:
        healpix_order = self.base_healpix_order

        base_quat = numpy_to_tensor(
            so3_grid.grid_SO3(healpix_order), device=self.device, dtype=torch.float32
        )
        self._set_buffer("base_quat", base_quat, persistent=False)

        base_rot = quaternion_to_matrix(base_quat)
        self._set_buffer("base_rot", base_rot, persistent=False)
        self.num_base_rot = len(base_rot)

        trans_healpix_order = self.base_trans_healpix_order
        # Translations are defined in pixel units on the input Fourier grid (D) and are
        # independent of the current frequency window (side_length L).
        base_trans = numpy_to_tensor(
            shift_grid.base_shift_grid(
                trans_healpix_order,
                self.trans_grid_extent,
                self.trans_grid_samples,
                xshift=self.trans_grid_x_shift,
                yshift=self.trans_grid_y_shift,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self._set_buffer("base_trans", base_trans, persistent=False)
        self.num_base_trans = len(base_trans)

    def _refresh_radial_buffers(self) -> None:
        with torch.no_grad():
            pixel2ring_idx = fftindex_radial2d(self.side_length, device=self.device)

            self.R = self.side_length // 2 + 1
            valid_pixel_mask = pixel2ring_idx < self.R
            self._set_buffer("valid_pixel_mask", valid_pixel_mask, persistent=False)
            self.P = int(valid_pixel_mask.sum().item())

            valid_pixel2ring_idx = pixel2ring_idx[valid_pixel_mask].contiguous()
            self._set_buffer("valid_pixel2ring_idx", valid_pixel2ring_idx, persistent=False)

            counts = torch.bincount(valid_pixel2ring_idx, minlength=self.R).float()
            denom = torch.zeros_like(counts)
            denom[counts > 0] = 1.0 / counts[counts > 0]
            self._set_buffer("ring_denom", denom, persistent=False)

    def _radial_residual_power(
        self,
        proj_image: torch.Tensor,
        trans_image: torch.Tensor,
        *,
        sel2proj_idx: torch.LongTensor,
        sel2trans_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute radial residual power for selected hypotheses.

        Args:
            proj_image: Projection buffer of shape ``(U_proj, P)`` or ``(B*K*Q, P)``.
            trans_image: Translated-image buffer of shape ``(U_trans, P)`` or ``(B*T, P)``.
            sel2proj_idx: Row indices mapping each selected hypothesis to ``proj_image``.
            sel2trans_idx: Row indices mapping each selected hypothesis to ``trans_image``.

        Returns:
            Tensor of shape ``(N_sel, R)`` containing the per-ring mean squared residual
            for each selected hypothesis.
        """
        sel2proj_idx = sel2proj_idx.to(device=proj_image.device, dtype=torch.long)
        sel2trans_idx = sel2trans_idx.to(device=trans_image.device, dtype=torch.long)
        ring_idx = self.valid_pixel2ring_idx
        ring_denom = self.ring_denom
        if ring_idx.device != proj_image.device:
            ring_idx = ring_idx.to(proj_image.device)
        if ring_denom.device != proj_image.device:
            ring_denom = ring_denom.to(proj_image.device)

        N = int(sel2proj_idx.numel())
        radial_residual_power = torch.zeros((N, int(self.R)), device=proj_image.device, dtype=torch.float32)
        residual_chunk = max(1, int(self.mse_chunk))
        for chunk_start in range(0, N, residual_chunk):
            chunk_end = min(chunk_start + residual_chunk, N)
            proj_sel = proj_image.index_select(0, sel2proj_idx[chunk_start:chunk_end])
            trans_sel = trans_image.index_select(0, sel2trans_idx[chunk_start:chunk_end])
            residual_power = (proj_sel - trans_sel).abs().square().to(torch.float32)
            radial_residual_power[chunk_start:chunk_end].scatter_add_(
                1,
                ring_idx.view(1, -1).expand(chunk_end - chunk_start, -1),
                residual_power,
            )
            radial_residual_power[chunk_start:chunk_end] *= ring_denom.view(1, -1)

        return radial_residual_power

    
    def _refresh_caches(self) -> None:
        dist = getattr(torch, "distributed", None)

        is_rank0 = True
        if dist is not None and dist.is_available() and dist.is_initialized():
            try:
                is_rank0 = dist.get_rank() == 0
            except Exception as e:
                is_rank0 = True
                warnings.warn(f"Failed to query distributed rank: {e}", RuntimeWarning)

        old_ssd_cache = getattr(self, "ssd_cache", None)
        old_ssd_dir = None if old_ssd_cache is None else old_ssd_cache.active_dir()

        if old_ssd_cache is not None:
            try:
                old_ssd_cache.flush()
            except Exception as e:
                if is_rank0:
                    warnings.warn(f"Failed to flush old SSD cache: {e}", RuntimeWarning)

        if dist is not None and dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                if is_rank0:
                    warnings.warn(f"Failed to enter distributed barrier (pre-cache-refresh): {e}", RuntimeWarning)

        self.memory_cache = MemoryProjCache()

        if self.ssd_cache_root is not None:
            volume_version = getattr(self.volume, "volume_version", None)
            if volume_version is None:
                volume_tag = "vol_none"
            else:
                try:
                    volume_tag = f"vol_{int(volume_version)}"
                except Exception:
                    volume_tag = f"vol_{str(volume_version)}"

            new_ssd_dir = os.path.abspath(
                os.path.join(
                    self.ssd_cache_root,
                    f"epoch_{self.state.progress.epoch}_half_{self.state.progress.half}_{volume_tag}",
                )
            )
            self.ssd_cache = SSDProjCache(cache_dir=new_ssd_dir)
        else:
            new_ssd_dir = None
            self.ssd_cache = SSDProjCache()

        if dist is not None and dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                if is_rank0:
                    warnings.warn(f"Failed to enter distributed barrier (post-cache-refresh): {e}", RuntimeWarning)

        if old_ssd_cache is not None and old_ssd_dir is not None and old_ssd_dir != new_ssd_dir:
            try:
                old_ssd_cache.close(
                    delete_root=is_rank0,
                    remove_locks=is_rank0,
                    async_clear=True,
                    cleanup=is_rank0,
                )
            except Exception as e:
                if is_rank0:
                    warnings.warn(f"Failed to close old SSD cache: {e}", RuntimeWarning)

    def refresh(self) -> None:
        self._refresh_schedule_state()
        self._refresh_pose_grid()
        self._refresh_caches()
        self._refresh_radial_buffers()
        self.volume_version = getattr(self.volume, "volume_version", None)

    def clear_memory_cache(self) -> None:
        if self.memory_cache is not None:
            self.memory_cache.clear()


    def _subdivide_rot_candidates(
        self,
        quat: torch.Tensor | np.ndarray,
        rot_grid_idx: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """Subdivide rotation candidates to the next HEALPix order.

        Given candidate quaternions and their ``(s2_idx, s1_idx)`` indices at the
        current HEALPix order ``self.current_healpix_order``, this returns the 8
        nearest SO(3) neighbors at the next order ``self.current_healpix_order + 1``.

        Args:
            quat: Quaternions with shape ``(N, 4)``.
            rot_grid_idx: HEALPix grid indices with shape ``(N, 2)`` at the current
                order. Each row is ``(s2_idx, s1_idx)``.

        Returns:
            quat: Neighbor quaternions with shape ``(8 * N, 4)`` at the next order.
            rot_grid_idx: Neighbor indices with shape ``(8 * N, 2)`` at the next order.
            rotmat: Rotation matrices for ``quat`` with shape ``(8 * N, 3, 3)``.
        """
        current_healpix_order = int(self.current_healpix_order)
        return so3_grid.subdivide_neighbors(
            quat,
            rot_grid_idx,
            current_healpix_order,
            device=self.device,
        )

    def _subdivide_trans_candidates(
        self,
        trans_grid_idx: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subdivide translation candidates to the next translation-grid resolution.

        The translation grid is defined at ``self.current_trans_healpix_order``.
        Given translation-grid indices ``(ix, iy)`` at that resolution, this returns the 4
        nearest neighbors at the next resolution level (one level finer).

        Args:
            trans_grid_idx: Translation grid indices with shape ``(N, 2)`` at order
                ``self.current_trans_healpix_order``.

        Returns:
            trans: Neighbor translation-grid coordinates with shape ``(4 * N, 2)``.
                These are grid-defined translations before adding any per-image
                ``pose.translation(...)`` center.
            trans_grid_idx: Neighbor indices with shape ``(4 * N, 2)``.
        """
        trans_healpix_order = int(self.current_trans_healpix_order)

        trans_grid_idx = torch.as_tensor(
            trans_grid_idx, device=self.device, dtype=torch.long
        )
        trans, trans_grid_idx = shift_grid.get_neighbor(
            trans_grid_idx[:, 0],
            trans_grid_idx[:, 1],
            trans_healpix_order,
            self.trans_grid_extent,
            self.trans_grid_samples,
        )

        trans = trans.reshape(-1, 2).to(self.device, dtype=torch.float32)
        trans_grid_idx = trans_grid_idx.reshape(-1, 2).to(
            self.device, dtype=torch.long
        )

        return trans, trans_grid_idx


    def _prepare_ctf_valid_pixels(
        self,
        ctf: torch.Tensor,
        *,
        device: torch.device,
        side_length: int,
        img_idx: torch.LongTensor | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        L = int(side_length)

        ctf = downsample2d(ctf.to(device), L)
        ctf = ctf.masked_fill(
            ~circular_mask(L, L, center=(L // 2, L // 2), device=device),
            0.0,
        )

        mask = self.valid_pixel_mask
        if mask.device != device:
            mask = mask.to(device)

        ctf = ctf.view(-1, L * L)[:, mask].contiguous()
        if img_idx is not None:
            ctf = ctf.index_select(0, img_idx.to(device))

        if dtype is not None:
            ctf = ctf.to(dtype=dtype)
        return ctf

    def _unique_with_first_index(
        self,
        x: torch.Tensor,
        *,
        dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

        unique_x, inverse = torch.unique(x, dim=dim, return_inverse=True)

        if inverse.numel() == 0:
            first_idx = torch.empty((0,), device=inverse.device, dtype=torch.long)
            return unique_x, inverse, first_idx

        idx = torch.arange(inverse.numel(), device=inverse.device, dtype=torch.long)
        first_idx = torch.full(
            (int(unique_x.shape[0]),),
            idx.numel(),
            dtype=torch.long,
            device=inverse.device,
        )
        first_idx.scatter_reduce_(0, inverse, idx, reduce="amin")
        return unique_x, inverse, first_idx

    def _project_global(
        self,
        rotation: torch.Tensor,
        *,
        ctf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project the 3D Fourier volume into 2D Fourier slices (central slices).

        Base projections (without CTF) are cached in memory and keyed by ``side_length``.

        Args:
            rotation: Rotation matrices of shape ``(Q, 3, 3)``.
            ctf: Optional CTF tensor of shape ``(B, D, D)``. If provided, the CTF is downsampled
                to ``(B, L, L)``, masked, and applied to the projections.

        Returns:
            Complex Fourier projections of shape ``(B, K * Q, P)``, where ``P`` is the number of
            valid Fourier pixels selected by ``valid_pixel_mask``.
        """
        B = int(ctf.shape[0]) if ctf is not None else 1
        K = int(self.volume.num_volumes)
        Q = int(rotation.shape[0])
        L = self.side_length

        cache_ok = self.memory_cache.base_proj_cache_ok(side_length=L)

        if not cache_ok:
            rotation = rotation.view(1, Q, 3, 3).expand(K, -1, -1, -1)  # (K, Q, 3, 3)

            if self.pose_chunk_factor is not None:
                proj_chunk = math.ceil((self.pose_chunk_factor / L) ** 2 / K)
                raw_proj = torch.empty(K, Q, L, L, dtype=torch.complex64, device=self.device)
                for chunk_start in range(0, rotation.shape[1], proj_chunk):
                    chunk_end = min(chunk_start + proj_chunk, rotation.shape[1])
                    raw_proj[:, chunk_start:chunk_end] = self.volume.project(
                        rotation[:, chunk_start:chunk_end], side_length=L
                    )
            else:
                raw_proj = self.volume.project(rotation, side_length=L)  # (K, Q, L, L)

            raw_proj = raw_proj.reshape(K * Q, L * L)[:, self.valid_pixel_mask].contiguous()  # (K * Q, P)
            self.memory_cache.set_base_proj(proj=raw_proj, side_length=L)
        else:
            raw_proj = self.memory_cache.get_base_proj()  # (K * Q, P)

        proj = raw_proj.unsqueeze(0)  # (1, K * Q, P)

        if ctf is not None:
            ctf_valid = self._prepare_ctf_valid_pixels(
                ctf,
                device=proj.device,
                side_length=L,
                dtype=raw_proj.real.dtype,
            )
            B = int(ctf_valid.shape[0])
            ctf_valid = ctf_valid.view(B, 1, -1).expand(-1, K * Q, -1)
            proj = proj * ctf_valid

        return proj
    
    def _translate_global(self, image: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """Apply 2D translations to Fourier-domain images.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex). Translations are
                applied on this full-resolution FFT grid and the result is then center-cropped
                to ``(B, L, L)`` using ``side_length``.
            translation: Translation vectors of shape ``(B, T, 2)`` in pixels on the input
                grid ``D`` (not scaled by ``side_length``), where each row is ``(dx, dy)``.

        Returns:
            Complex tensor of translated images with shape ``(B, T, P)``, where ``P`` is the
            number of valid Fourier pixels selected by ``valid_pixel_mask``.
        """
        B = int(image.shape[0])
        if translation.ndim != 3 or int(translation.shape[0]) != B or int(translation.shape[-1]) != 2:
            raise ValueError(
                f"translation must have shape (B, T, 2) with B={B}, got {tuple(translation.shape)}"
            )
        T = int(translation.shape[1])
        D = int(image.shape[-1])
        L = int(self.side_length)

        translation = translation.to(device=image.device)

        # NOTE:
        # - translate_image interprets (dx, dy) in pixel units of the FFT grid of its input.
        # - We translate on the original D×D grid (for correct units) and then crop to L.
        if self.pose_chunk_factor is not None:
            trans_chunk = math.ceil((self.pose_chunk_factor / D) ** 2)
            trans_image = torch.empty(B, T, D, D, dtype=image.dtype, device=image.device)
            for chunk_start in range(0, T, trans_chunk):
                chunk_end = min(chunk_start + trans_chunk, T)
                trans = translation[:, chunk_start:chunk_end]
                trans_image[:, chunk_start:chunk_end] = translate_image(image, trans)
        else:
            trans_image = translate_image(image, translation)  # (B, T, D, D)

        trans_image = downsample2d(trans_image, L)  # (B, T, L, L)

        mask = self.valid_pixel_mask
        if mask.device != trans_image.device:
            mask = mask.to(trans_image.device)
        return trans_image.reshape(B, T, L * L)[:, :, mask].contiguous()  # (B, T, P)
    
    def _get_raw_proj_per_unique_rot(
        self,
        *,
        rotmat: torch.Tensor,
        unique_rot_id: torch.LongTensor,
        unique_rot_first_idx: torch.LongTensor,
        healpix_order: int,
        side_length: int,
    ) -> torch.Tensor:
        K = int(self.volume.num_volumes)
        L = int(side_length)
        proj_cache_backend = self.state.schedule.proj_cache_backend

        rotmat_unique = rotmat[unique_rot_first_idx].view(1, -1, 3, 3).expand(
            K, -1, -1, -1
        )  # (K, U_rot, 3, 3)
        U_rot = int(rotmat_unique.shape[1])

        if proj_cache_backend == "none":
            if self.pose_chunk_factor is not None:
                proj_chunk = math.ceil((self.pose_chunk_factor / L) ** 2 / K)
                raw_proj_per_uniq_rot = torch.empty(
                    K, U_rot, L, L, dtype=torch.complex64, device=self.device
                )
                for chunk_start in range(0, U_rot, proj_chunk):
                    chunk_end = min(chunk_start + proj_chunk, U_rot)
                    raw_proj_per_uniq_rot[:, chunk_start:chunk_end] = self.volume.project(
                        rotmat_unique[:, chunk_start:chunk_end], side_length=L
                    )
            else:
                raw_proj_per_uniq_rot = self.volume.project(rotmat_unique, side_length=L)

            mask = self.valid_pixel_mask
            if mask.device != raw_proj_per_uniq_rot.device:
                mask = mask.to(raw_proj_per_uniq_rot.device)
            return raw_proj_per_uniq_rot.reshape(K, U_rot, L * L)[:, :, mask].contiguous()

        unique_rot_id_cpu = unique_rot_id.detach().cpu().tolist()
        unique_keys = [
            pack_proj_cache_key(uid=uid, healpix_order=int(healpix_order), side_length=L)
            for uid in unique_rot_id_cpu
        ]

        if proj_cache_backend == "ssd":
            if self.ssd_cache.active_dir() is None:
                raise ValueError("SSD cache is not active.")
            missing_positions = self.ssd_cache.find_missing_positions(unique_keys)
        elif proj_cache_backend == "memory":
            missing_positions = self.memory_cache.find_missing_positions(unique_keys)
        else:
            raise ValueError(f"Unknown proj_cache_backend: {proj_cache_backend}")

        raw_proj_missing = None
        if missing_positions:
            missing_pos_t = torch.as_tensor(
                missing_positions, dtype=torch.long, device=unique_rot_first_idx.device
            )
            unique_idx_missing = unique_rot_first_idx[missing_pos_t]

            rotmat_missing = rotmat[unique_idx_missing].view(1, -1, 3, 3).expand(
                K, -1, -1, -1
            )  # (K, U_miss, 3, 3)
            U_miss = int(rotmat_missing.shape[1])

            if self.pose_chunk_factor is not None:
                proj_chunk = math.ceil((self.pose_chunk_factor / L) ** 2 / K)
                raw_proj_missing = torch.empty(
                    K, U_miss, L, L, dtype=torch.complex64, device=self.device
                )
                for chunk_start in range(0, U_miss, proj_chunk):
                    chunk_end = min(chunk_start + proj_chunk, U_miss)
                    raw_proj_missing[:, chunk_start:chunk_end] = self.volume.project(
                        rotmat_missing[:, chunk_start:chunk_end], side_length=L
                    )
            else:
                raw_proj_missing = self.volume.project(rotmat_missing, side_length=L)

            raw_proj_missing = raw_proj_missing.detach()
            mask = self.valid_pixel_mask
            if mask.device != raw_proj_missing.device:
                mask = mask.to(raw_proj_missing.device)
            raw_proj_missing = raw_proj_missing.reshape(K, U_miss, L * L)[:, :, mask].contiguous()

            if proj_cache_backend == "memory":
                for ii, jj in enumerate(missing_positions):
                    self.memory_cache.put(
                        unique_keys[jj], raw_proj_missing[:, ii, ...], store_on_cpu=True
                    )

        if proj_cache_backend == "ssd":
            raw_proj_per_uniq_rot = torch.empty(
                (K, len(unique_keys), self.P),
                dtype=torch.complex64,
                device=self.device,
            )

            missing_pos_set = set(missing_positions)
            if missing_positions:
                missing_pos_t = torch.as_tensor(
                    missing_positions, dtype=torch.long, device=self.device
                )
                raw_proj_per_uniq_rot.index_copy_(1, missing_pos_t, raw_proj_missing)

            for pos, uk in enumerate(unique_keys):
                if pos in missing_pos_set:
                    continue
                raw_proj_per_uniq_rot[:, pos, ...] = self.ssd_cache.get(
                    uk, device=self.device
                )

            if missing_positions:
                for pos in missing_positions:
                    self.ssd_cache.put(
                        unique_keys[pos], raw_proj_per_uniq_rot[:, pos, ...], async_write=True
                    )

        elif proj_cache_backend == "memory":
            raw_proj_per_uniq_rot = self.memory_cache.stack_many(
                unique_keys, device=self.device, dim=1
            )

        else:
            raise ValueError(f"Unknown proj_cache_backend: {proj_cache_backend}")

        return raw_proj_per_uniq_rot

    def _project_oversampling(
        self,
        rot_grid_idx: torch.LongTensor,
        rotmat: torch.Tensor,
        rot2img_idx: torch.LongTensor,
        rot2vol_idx: torch.LongTensor,
        *,
        ctf: torch.Tensor | None = None,
    ):
        """Project volumes for an oversampled SO(3) grid with caching.

        In the oversampling stage each candidate rotation is associated with a specific
        volume (via ``rot2vol_idx``). This function projects all volumes for the unique
        rotations (optionally using the projection cache), and then selects the requested
        volume per unique ``(image, volume, rotation)`` triplet.

        Args:
            rot_grid_idx: HEALPix grid indices of shape ``(Q, 2)``. Each row is
                ``(s2_idx, s1_idx)`` at ``self.current_healpix_order``.
            rotmat: Rotation matrices of shape ``(Q, 3, 3)`` corresponding to
                ``rot_grid_idx``.
            rot2img_idx: Image indices of shape ``(Q,)`` mapping each candidate rotation
                to its image.
            rot2vol_idx: Volume indices of shape ``(Q,)`` mapping each candidate rotation
                to a volume id in ``[0, K)``.
            ctf: Optional per-image CTF of shape ``(B, D, D)`` or ``(B, L, L)``.
                When given, it is downsampled to ``(B, L, L)``, masked by
                ``valid_pixel_mask``, and applied to the projections.

        Returns:
            proj_per_uniq_req: Projections of shape ``(U, P)``, where ``U`` is the number
                of unique ``(image, volume, rotation)`` requests and ``P`` is the number
                of valid Fourier pixels.
            unique_req_inverse: Inverse map of shape ``(Q,)`` such that
                ``proj_per_uniq_req[unique_req_inverse]`` expands back to the
                per-candidate ordering.
            rot_id: Flattened rotation ids of shape ``(Q,)`` computed as
                ``s2_idx * (6 * 2**order) + s1_idx``.
            unique_req_first_idx: First-occurrence indices of shape ``(U,)`` into the
                original candidate list for each unique request.
        """
        L = int(self.side_length)

        order = int(self.current_healpix_order)
        n_pix_s1 = int(6 * (2**order))

        # rot_grid_idx[:, 0] = s2 index, rot_grid_idx[:, 1] = s1 index
        # flatten (s2i, s1i) -> rot_id
        rot_id = rot_grid_idx[:, 0] * n_pix_s1 + rot_grid_idx[:, 1]  # (Q,)
        unique_rot_id, rot_inverse, unique_rot_first_idx = self._unique_with_first_index(rot_id)

        req_id = torch.stack([rot2img_idx, rot2vol_idx, rot_id], dim=1)  # (Q, 3)
        unique_req_id, unique_req_inverse, unique_req_first_idx = self._unique_with_first_index(
            req_id, dim=0
        )

        uniq_req_idx = torch.arange(unique_req_id.shape[0], device=unique_req_id.device)
        uniq_req2img_idx = unique_req_id[:, 0]  # (U,)
        uniq_req2vol_idx = unique_req_id[:, 1]  # (U,)
        uniq_req2uniq_rot_idx = rot_inverse[unique_req_first_idx]  # (U,)

        raw_proj_per_uniq_rot = self._get_raw_proj_per_unique_rot(
            rotmat=rotmat,
            unique_rot_id=unique_rot_id,
            unique_rot_first_idx=unique_rot_first_idx,
            healpix_order=order,
            side_length=L,
        )
        proj_per_uniq_req = raw_proj_per_uniq_rot.index_select(1, uniq_req2uniq_rot_idx)[uniq_req2vol_idx, uniq_req_idx, :] # (U, P)

        if ctf is not None:
            ctf_valid = self._prepare_ctf_valid_pixels(
                ctf,
                device=proj_per_uniq_req.device,
                side_length=L,
                img_idx=uniq_req2img_idx,
                dtype=proj_per_uniq_req.real.dtype,
            )
            proj_per_uniq_req = proj_per_uniq_req * ctf_valid

        return proj_per_uniq_req, unique_req_inverse, rot_id, unique_req_first_idx
    

    def _translate_indexed(
        self,
        image: torch.Tensor,
        trans_grid_idx: torch.LongTensor,
        trans: torch.Tensor,
        trans2img_idx: torch.LongTensor,
    ):
        """Translate images for a 2D translation grid with de-duplication.
 
        In the oversampling stage, translations are refined around previously selected
        candidates. Translation only depends on the image, so requests are de-duplicated
        across the batch by the pair ``(trans2img_idx[i], trans_grid_idx[i])``.
 

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex). Translations are
                applied on this full-resolution FFT grid and the result is then center-cropped
                to ``(B, L, L)`` using ``side_length``.
            trans_grid_idx: Translation grid indices of shape ``(T, 2)``. Each row is
                ``(ix, iy)`` at the current translation-grid resolution.
            trans: Translation vectors of shape ``(T, 2)`` in pixels on the input grid ``D``
                (not scaled by ``side_length``), where each row is ``(dx, dy)``.
            trans2img_idx: Image indices of shape ``(T,)`` mapping each translation candidate
                to its image.

        Returns:
            trans_per_uniq_req: Translated images of shape ``(U, P)``, where ``U`` is the
                number of unique ``(image, translation)`` pairs and ``P`` is the number of
                valid Fourier pixels.
            unique_req_inverse: Inverse map of shape ``(T,)`` such that
                ``trans_per_uniq_req[unique_req_inverse]`` expands back to per-candidate
                ordering.
            trans_id: Flattened translation ids of shape ``(T,)`` computed as
                ``ix * n_pix_1d + iy``.
            unique_req_first_idx: First-occurrence indices of shape ``(U,)`` into the original
                candidate list for each unique request.
        """
        D = int(image.shape[-1])
        L = int(self.side_length)
        device = image.device

        trans_grid_idx = trans_grid_idx.to(device=device)
        trans2img_idx = trans2img_idx.to(device=device)
        trans = trans.to(device=device)

        n_pix_1d = int(self.trans_grid_samples * (2 ** self.current_trans_healpix_order))
        trans_id = trans_grid_idx[:, 0] * n_pix_1d + trans_grid_idx[:, 1]  # (T,)

        req_id = torch.stack([trans2img_idx, trans_id], dim=1)  # (T, 2)
        unique_req_id, unique_req_inverse, unique_req_first_idx = self._unique_with_first_index(
            req_id, dim=0
        )

        uniq_req2img_idx = unique_req_id[:, 0]
        image_per_uniq_req = image.index_select(0, uniq_req2img_idx)
        trans_per_uniq_req = trans.index_select(0, unique_req_first_idx)
        U = int(trans_per_uniq_req.shape[0])

        if self.pose_chunk_factor is not None:
            trans_chunk = math.ceil((self.pose_chunk_factor / D) ** 2)
            trans_image_per_uniq_req = torch.empty(U, D, D, dtype=image.dtype, device=device)
            for chunk_start in range(0, U, trans_chunk):
                chunk_end = min(chunk_start + trans_chunk, U)
                trans_image_per_uniq_req[chunk_start:chunk_end] = translate_image(
                    image_per_uniq_req[chunk_start:chunk_end],
                    trans_per_uniq_req[chunk_start:chunk_end],
                )
        else:
            trans_image_per_uniq_req = translate_image(image_per_uniq_req, trans_per_uniq_req)  # (U, D, D)

        # NOTE:
        # - translate_image interprets (dx, dy) in pixel units of the FFT grid of its input.
        # - `trans` is always in pixels of the original input grid D (search() unit convention).
        # - We therefore translate on D×D and then crop to side_length L for likelihood eval.
        trans_image_per_uniq_req = downsample2d(trans_image_per_uniq_req, L)

        mask = self.valid_pixel_mask
        if mask.device != trans_image_per_uniq_req.device:
            mask = mask.to(trans_image_per_uniq_req.device)
        trans_image_per_uniq_req = trans_image_per_uniq_req.reshape(U, L * L)[:, mask].contiguous()  # (U, P)

        return trans_image_per_uniq_req, unique_req_inverse, trans_id, unique_req_first_idx
    
    
    def _unique_hypo_idx_oversampling(
        self,
        *,
        sel2img_idx: torch.LongTensor,
        sel2vol_idx: torch.LongTensor,
        rot_id: torch.LongTensor,
        trans_id: torch.LongTensor,
        unique_proj_req_inverse: torch.LongTensor,
        unique_trans_req_inverse: torch.LongTensor,
    ):
        """Assemble unique hypotheses for a single oversampling refinement round.

        Each selected hypothesis spawns a fixed set of refined candidates:

        - 8 rotation neighbors (at the next HEALPix order)
        - 4 translation neighbors (at the next translation-grid resolution)

        During oversampling, each rotation candidate is already tied to a specific volume
        via ``sel2vol_idx``. This function constructs the implicit combinations and
        optionally de-duplicates them using the hypothesis key
        ``(img_idx, vol_idx, rot_id, trans_id)``.
        """
        N_sel = sel2img_idx.numel()
        if N_sel == 0:
            empty = torch.empty((0,), device=sel2img_idx.device, dtype=torch.long)
            return empty, empty, empty, empty

        if rot_id.numel() % N_sel != 0:
            raise ValueError(
                f"rot_id.numel()={rot_id.numel()} is not divisible by N_sel={N_sel}"
            )
        if trans_id.numel() % N_sel != 0:
            raise ValueError(
                f"trans_id.numel()={trans_id.numel()} is not divisible by N_sel={N_sel}"
            )

        Q = rot_id.numel() // N_sel
        T = trans_id.numel() // N_sel

        if Q != 8:
            raise ValueError(f"Expected Q=8, but got {Q}")
        if T != 4:
            raise ValueError(f"Expected T=4, but got {T}")

        device = sel2img_idx.device

        hypo2img_idx = sel2img_idx.view(N_sel, 1, 1).expand(-1, Q, T).reshape(-1)
        hypo2vol_idx = sel2vol_idx.view(N_sel, 1, 1).expand(-1, Q, T).reshape(-1)
        hypo2rot_idx = torch.arange(N_sel * Q, device=device, dtype=torch.long).view(N_sel, Q, 1).expand(-1, -1, T).reshape(-1)
        hypo2trans_idx = torch.arange(N_sel * T, device=device, dtype=torch.long).view(N_sel, 1, T).expand(-1, Q, -1).reshape(-1)

        rot_id_per_hypo = rot_id[hypo2rot_idx]
        trans_id_per_hypo = trans_id[hypo2trans_idx]
        hypo2proj_req_idx = unique_proj_req_inverse[hypo2rot_idx]
        hypo2trans_req_idx = unique_trans_req_inverse[hypo2trans_idx]

        if self.oversampling_deduplicate:
            hypo_id = torch.stack(
                [hypo2img_idx, hypo2vol_idx, rot_id_per_hypo, trans_id_per_hypo], dim=1
            )
            _, _, unique_hypo_first_idx = self._unique_with_first_index(
                hypo_id, dim=0
            )

            uniq_hypo2img_idx = hypo2img_idx[unique_hypo_first_idx]
            uniq_hypo2vol_idx = hypo2vol_idx[unique_hypo_first_idx]
            uniq_hypo2proj_req_idx = hypo2proj_req_idx[unique_hypo_first_idx]
            uniq_hypo2trans_req_idx = hypo2trans_req_idx[unique_hypo_first_idx]
        else:
            uniq_hypo2img_idx = hypo2img_idx
            uniq_hypo2vol_idx = hypo2vol_idx
            uniq_hypo2proj_req_idx = hypo2proj_req_idx
            uniq_hypo2trans_req_idx = hypo2trans_req_idx

        order = torch.argsort(uniq_hypo2img_idx)
        uniq_hypo2img_idx = uniq_hypo2img_idx[order]
        uniq_hypo2vol_idx = uniq_hypo2vol_idx[order]
        uniq_hypo2proj_req_idx = uniq_hypo2proj_req_idx[order]
        uniq_hypo2trans_req_idx = uniq_hypo2trans_req_idx[order]

        return uniq_hypo2img_idx, uniq_hypo2vol_idx, uniq_hypo2proj_req_idx, uniq_hypo2trans_req_idx
    

    def _evaluate_broadcast(
        self,
        proj_image: torch.Tensor,
        trans_image: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-image probabilities over a broadcast hypothesis grid.

        This evaluates the Cartesian product grid over ``(volume, rotation, translation)``
        by computing a weighted spectral MSE between projected volumes and translated
        images, and converting it to probabilities via ``exp(-mse)``.

        Distributed evaluation:
            If ``torch.distributed`` is initialized and ``device_mesh`` is provided,
            the computation is sharded across the projection-hypothesis dimension
            ``KQ``. Each rank in ``device_mesh.get_group(1)`` computes MSE for a
            contiguous slice of ``KQ`` (writing into the corresponding slice of the
            output tensor) and then performs an ``all_reduce`` sum so that all ranks
            obtain identical full results.

        Args:
            proj_image: Complex projections with shape ``(B, KQ, P)``.
            trans_image: Complex translated images with shape ``(B, T, P)``.

        Returns:
            Flattened probabilities with shape ``(B * KQ * T,)``. Probabilities are
            normalized independently for each image.
        """

        B = int(trans_image.shape[0])
        KQ = int(proj_image.shape[1])
        T = int(trans_image.shape[1])
 
        if int(proj_image.shape[0]) != B:
            raise ValueError(
                f"proj_image batch dim must match trans_image: {int(proj_image.shape[0])} vs {B}"
            )

        @torch.no_grad()
        def compute_mse_broadcast(proj_image, trans_image):
            device = self.device
            R = self.R
            if self.noise is None:
                precision = torch.ones((R,), device=device, dtype=torch.float32)
            else:
                precision = self.noise.precision[:R].to(device)
            finite = torch.isfinite(precision)

            weight_r = torch.zeros_like(precision)
            weight_r[finite] = 0.5 * precision[finite]
            weight_r[0] = 0
            if self.ring_averaged_mse:
                weight_r = weight_r * self.ring_denom

            weight = weight_r[self.valid_pixel2ring_idx].contiguous()  # (P,)


            if proj_image.dim() != 3 or trans_image.dim() != 3:
                raise ValueError(
                    f"expected proj_image/trans_image to be flattened (B,*,P); got {tuple(proj_image.shape)} and {tuple(trans_image.shape)}"
                )

            mse = torch.zeros((B, KQ, T), device=self.device, dtype=torch.float32)

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

            slice_size = math.ceil(KQ / calculation_parallel_size)

            slice_start = slice_size * calculation_process_rank
            slice_end = min(slice_size * (calculation_process_rank + 1), KQ)

            if slice_start < slice_end:
                mse_chunk = self.mse_chunk
                for chunk_start in range(slice_start, slice_end, mse_chunk):
                    chunk_end = min(chunk_start + mse_chunk, slice_end)
                    spectral_mse_loss(
                        proj_image[:, chunk_start:chunk_end],
                        trans_image,
                        weight=weight,
                        out=mse[:, chunk_start:chunk_end, :],
                        reduction="none",
                    )

            if (
                dist is not None
                and dist.is_available()
                and dist.is_initialized()
                and calculation_parallel_size > 1
            ):
                dist.all_reduce(mse, dist.ReduceOp.SUM, group)

            return mse.view(B, -1)

        mse = compute_mse_broadcast(proj_image, trans_image) # (B, K * Q * T)

        # Tackle underflow in exp()
        mse_min = torch.amin(mse, dim=-1, keepdim=True) # (B, 1)
        mse = mse - mse_min
        prob = torch.exp(-mse) # (B, K * Q * T)
        prob /= prob.sum(dim=-1, keepdim=True)
        prob = prob.reshape(-1) # (B * K * Q * T,)

        return prob

    def _evaluate_indexed(
        self,
        proj_image: torch.Tensor,
        trans_image: torch.Tensor,
        num_images: int,
        hypo2img_idx: torch.LongTensor,
        hypo2proj_idx: torch.LongTensor,
        hypo2trans_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute per-hypothesis probabilities for an indexed hypothesis set.
         
        This method evaluates only the hypotheses specified by index vectors.
        It is used for oversampling rounds after request de-duplication.
 
        Numerical stability:
            Probabilities are computed as ``exp(-mse)``. To reduce underflow, the
            per-image minimum MSE is subtracted before exponentiation.
 
        Distributed evaluation:
            If ``torch.distributed`` is initialized and ``device_mesh`` is
            provided, hypotheses are sharded across the calculation group
            ``device_mesh.get_group(1)``. Each rank computes MSE for a contiguous
            slice of hypotheses, then an ``all_reduce`` sum assembles the full
            MSE vector on every rank.
 
        Args:
            proj_image: Complex projections with shape ``(U_proj, P)``.
            trans_image: Complex translated images with shape ``(U_trans, P)``.
            num_images: Number of images ``B`` used for per-image normalization.
            hypo2img_idx: Image index for each hypothesis, shape ``(U_hypo,)``.
            hypo2proj_idx: Projection index into ``proj_image``, shape ``(U_hypo,)``.
            hypo2trans_idx: Translation index into ``trans_image``, shape ``(U_hypo,)``.
 
        Returns:
            Probabilities with shape ``(U_hypo,)``. For each image, probabilities
            over that image's hypotheses sum to 1.
        """

        B = int(num_images)
        U_hypo = int(hypo2img_idx.shape[0])
        
        @torch.no_grad()
        def compute_mse_indexed(proj_image, trans_image, hypo2proj_idx, hypo2trans_idx):  
            # f_proj: (K * U_proj, P)
            # f_trans: (U_trans, P)

            device = self.device
            R = self.R
            if self.noise is None:
                precision = torch.ones((R,), device=device, dtype=torch.float32)
            else:
                precision = self.noise.precision[:R].to(device)
            finite = torch.isfinite(precision)

            weight_r = torch.zeros_like(precision)
            weight_r[finite] = 0.5 * precision[finite]
            weight_r[0] = 0
            if self.ring_averaged_mse:
                weight_r = weight_r * self.ring_denom

            weight = weight_r[self.valid_pixel2ring_idx].contiguous()  # (P,)

            mse = torch.zeros(U_hypo, device=self.device)

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

            slice_size = math.ceil(U_hypo / calculation_parallel_size)
            chunk_start = calculation_process_rank * slice_size
            chunk_end = min((calculation_process_rank + 1) * slice_size, U_hypo)

            if chunk_start < chunk_end:
                out_flat = mse[chunk_start:chunk_end]
                spectral_mse_loss(
                    proj_image,
                    trans_image,
                    weight=weight,
                    input_indices=hypo2proj_idx[chunk_start:chunk_end],
                    target_indices=hypo2trans_idx[chunk_start:chunk_end],
                    out=out_flat,
                    reduction="none",
                )

            if (
                dist is not None
                and dist.is_available()
                and dist.is_initialized()
                and calculation_parallel_size > 1
            ):
                dist.all_reduce(mse, dist.ReduceOp.SUM, group)
            
            return mse

        mse = compute_mse_indexed(proj_image, trans_image, hypo2proj_idx, hypo2trans_idx) # (U_hypo,)

        # Subtract the min mse per image to prevent underflow 
        min_per_img = torch.full((B,), float('inf'), device=mse.device)
        min_per_img = min_per_img.scatter_reduce(0, hypo2img_idx, mse, reduce="amin") # (B,)
        
        mse = mse - min_per_img[hypo2img_idx]
        prob = torch.exp(-mse)
        sum_per_img =  torch.zeros(B, device=mse.device).scatter_add(0, hypo2img_idx, prob) # (B,)
        prob /= sum_per_img[hypo2img_idx] # (U_hypo,)

        return prob
        
    def _select_by_prob(
        self,
        hypo_prob: torch.Tensor,
        hypo2img_idx: torch.LongTensor,
        payload: dict[str, torch.Tensor],
    ):
        """Select top hypotheses per image by cumulative probability.

        Args:
            hypo_prob: Hypothesis probabilities of shape (N,).
            hypo2img_idx: Image index of each hypothesis, shape (N,).
            payload: Extra tensors aligned with hypotheses, each of shape (N,).
                These tensors are gathered together with the selected hypotheses.
                Examples:
                    global: {"vol": hypo2vol_idx, "rot": hypo2rot_idx, "trans": hypo2trans_idx}
                    oversampling: {"vol": hypo2vol_idx, "proj_req": hypo2proj_req_idx, "trans_req": hypo2trans_req_idx}

        Returns:
            sel_prob: Selected probabilities, shape (N_sel,).
            sel2img_idx: Image indices of selected hypotheses, shape (N_sel,).
            sel_payload: Dict with the same keys as ``payload``, each tensor of shape (N_sel,).
        """
        threshold = self.candidate_select_threshold
        max_candidates = self.max_candidates

        if hypo_prob.numel() == 0:
            raise ValueError("hypo_prob must be non-empty")
        if not torch.all(hypo2img_idx[:-1] <= hypo2img_idx[1:]):
            raise ValueError("hypo2img_idx must be sorted")

        if not torch.isfinite(hypo_prob).all():
            raise RuntimeError("hypo_prob contains non-finite values")

        N_hypo = hypo_prob.numel()
        for name, tensor in payload.items():
            if tensor.numel() != N_hypo:
                raise ValueError(
                    f"payload['{name}'] must have shape ({N_hypo},), got {tuple(tensor.shape)}"
                )

        img_idx_unique, img_counts = torch.unique_consecutive(hypo2img_idx, return_counts=True)
        img_idx_unique = img_idx_unique.detach().cpu().tolist()
        img_counts = img_counts.detach().cpu().tolist()

        sel_prob_parts = []
        sel2img_idx_parts = []
        sel_payload_parts = {name: [] for name in payload}

        start = 0
        for img_idx, img_cnt in zip(img_idx_unique, img_counts):
            end = start + img_cnt

            prob_sorted, prob_idx_sorted = torch.sort(hypo_prob[start:end], descending=True)
            prob_cumsum = torch.cumsum(prob_sorted, dim=0)

            keep_count = int((prob_cumsum < threshold).sum().item()) + 1
            keep_count = min(keep_count, img_cnt)
            if max_candidates is not None:
                keep_count = min(keep_count, max_candidates)

            sel_idx = start + prob_idx_sorted[:keep_count]

            sel_prob_part = prob_sorted[:keep_count].clone()
            if self.renormalize_sel_prob:
                sel_prob_part = sel_prob_part / sel_prob_part.sum()

            if LOGGER.isEnabledFor(logging.DEBUG):
                selected_ratio = 100.0 * keep_count / img_cnt
                top1 = float(prob_sorted[0].item())
                cutoff = float(prob_cumsum[keep_count - 1].item())
                LOGGER.debug(
                    "select_by_prob | candidates=%d/%d (%.2f%%) top1=%.4f cutoff=%.4f",
                    keep_count,
                    img_cnt,
                    selected_ratio,
                    top1,
                    cutoff,
                )

            sel_prob_parts.append(sel_prob_part)
            sel2img_idx_parts.append(
                torch.full(
                    (keep_count,),
                    img_idx,
                    device=hypo_prob.device,
                    dtype=torch.long,
                )
            )

            for name, tensor in payload.items():
                sel_payload_parts[name].append(tensor[sel_idx])

            start = end

        sel_prob = torch.cat(sel_prob_parts, dim=0)
        sel2img_idx = torch.cat(sel2img_idx_parts, dim=0)
        sel_payload = {
            name: torch.cat(parts, dim=0) for name, parts in sel_payload_parts.items()
        }

        return sel_prob, sel2img_idx, sel_payload

    def _gather_selected_candidates(
        self,
        *,
        quat: torch.Tensor,
        rot_grid_idx: torch.LongTensor,
        trans: torch.Tensor,
        trans_grid_idx: torch.LongTensor,
        unique_proj_req_first_idx: torch.LongTensor,
        unique_trans_req_first_idx: torch.LongTensor,
        sel2proj_req_idx: torch.LongTensor,
        sel2trans_req_idx: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.LongTensor]:
        sel_quat = quat[unique_proj_req_first_idx][sel2proj_req_idx]
        sel2rot_grid_idx = rot_grid_idx[unique_proj_req_first_idx][sel2proj_req_idx]

        sel_trans = trans[unique_trans_req_first_idx][sel2trans_req_idx]
        sel2trans_grid_idx = trans_grid_idx[unique_trans_req_first_idx][sel2trans_req_idx]

        return sel_quat, sel2rot_grid_idx, sel_trans, sel2trans_grid_idx

    def _select_best_per_image(
        self,
        *,
        B: int,
        sel2img_idx: torch.LongTensor,
        sel_quat: torch.Tensor,
        sel_trans: torch.Tensor,
    ):
        # Assumes selected candidates are grouped by image index, and within each image
        # group the first candidate is already the highest-probability one.
        best_quat = torch.zeros((B, 4), device=self.device, dtype=torch.float32)
        best_trans = torch.zeros((B, 2), device=self.device, dtype=torch.float32)

        sel_img_idx_unique, sel_img_counts = torch.unique_consecutive(
            sel2img_idx, return_counts=True
        )

        if sel_img_idx_unique.numel() != B:
            raise RuntimeError(
                f"Missing hypotheses for some images: "
                f"{sel_img_idx_unique.numel()} / {B}"
            )

        expected = torch.arange(
            B, device=sel_img_idx_unique.device, dtype=sel_img_idx_unique.dtype
        )
        if not torch.equal(sel_img_idx_unique, expected):
            raise RuntimeError(
                f"sel2img_idx must cover all images in order 0..{B - 1}, got {sel_img_idx_unique.tolist()}"
            )

        sel_best_src_idx = torch.cumsum(sel_img_counts, dim=0) - sel_img_counts

        best_quat.index_copy_(0, sel_img_idx_unique, sel_quat[sel_best_src_idx])
        best_trans.index_copy_(0, sel_img_idx_unique, sel_trans[sel_best_src_idx])

        return best_quat, best_trans

    @torch.no_grad()
    def search(
        self,
        image: torch.Tensor,
        *,
        particle_index: torch.LongTensor,
        ctf: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run HEALPix global pose search with optional oversampling refinement.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex).
            particle_index: Particle indices of shape ``(B,)``. Used when accumulating the
                best pose back into ``self.pose``.
            ctf: Optional per-image CTF tensor of shape ``(B, D, D)`` (or ``(B, L, L)``).
                The batch dimension must match ``image.shape[0]`` (no broadcasting).

        Returns:
            A 6-tuple ``(sel_prob, sel2img_idx, sel2vol_idx, sel_rotmat, sel_trans,
            sel_radial_residual_power)``, where:

            - sel_prob (torch.Tensor): Selected hypothesis probabilities with shape ``(N_sel,)``.
              Candidates are grouped by image index (``sel2img_idx``).
            - sel2img_idx (torch.LongTensor): Image indices for each selected hypothesis with shape
              ``(N_sel,)``.
            - sel2vol_idx (torch.LongTensor): Volume/class indices for each selected hypothesis with shape
              ``(N_sel,)``.
            - sel_rotmat (torch.Tensor): Rotation matrices for each selected hypothesis with shape
              ``(N_sel, 3, 3)``.
            - sel_trans (torch.Tensor): 2D translations (in pixels) for each selected hypothesis with shape
              ``(N_sel, 2)``.
            - sel_radial_residual_power (torch.Tensor | None): Optional per-hypothesis radial
              residual power with shape ``(N_sel, side_length // 2 + 1)``. Returned only when
              noise estimation is enabled.

              Unit convention: translations are expressed in pixels of the *input* Fourier grid
              (``D = image.shape[-1]``) and are not scaled by the current ``side_length``.
        """
        device = self.device

        image = image.to(device)
        particle_index = particle_index.to(device=device)

        B = int(image.shape[0])
        K = int(self.volume.num_volumes)
        pose_trans = self.pose.translation(particle_index).to(device=device, dtype=self.base_trans.dtype)
        # NOTE:
        # - All translations produced/consumed by this searcher are in pixel units of the input FFT grid
        #   (D = image.shape[-1]) and are never rescaled when side_length changes.
        # - Likelihood is evaluated on a cropped side_length-L window, but translations are applied on D×D
        #   first (via translate_image) and only then center-cropped to L.

        if ctf is not None:
            if not isinstance(ctf, torch.Tensor):
                ctf = torch.as_tensor(ctf)

            if ctf.ndim == 2:
                ctf = ctf.unsqueeze(0)

            if ctf.ndim != 3:
                raise ValueError(
                    f"ctf must have shape (B, D, D) or (B, L, L), got {tuple(ctf.shape)}"
                )

            if int(ctf.shape[0]) != B:
                raise ValueError(
                    f"ctf batch must match image batch: expected B={B}, got ctf.shape[0]={int(ctf.shape[0])} "
                    f"with full shape {tuple(ctf.shape)}"
                )

            ctf = ctf.to(device)

        self.current_healpix_order = self.base_healpix_order
        self.current_trans_healpix_order = self.base_trans_healpix_order
        sel_radial_residual_power = None
        proj_flat = None
        trans_flat = None
        sel2proj_flat_idx = None
        sel2trans_flat_idx = None

        current_volume_version = getattr(self.volume, "volume_version", None)
        if current_volume_version is None:
            raise RuntimeError(
                "HEALPix pose search requires volume.volume_version for cache validation, "
                "but it is missing."
            )

        if self.volume_version is None:
            self.volume_version = current_volume_version
        elif self.volume_version != current_volume_version:
            self.volume_version = current_volume_version
            self._refresh_caches()

        if self.state.schedule.pose_search_scope == "global":
            # Global search evaluates a coarse grid over:
            #   volume x SO(3) rotation x 2D translation.
            #
            # Projections are computed for all volumes and base rotations (optionally cached).
            # For each image, translation candidates are centered at its current pose
            # translation and offset by the shared base translation grid. Probabilities are
            # normalized per image and then truncated by cumulative probability to keep a
            # manageable candidate set.
            Q = self.num_base_rot
            T = self.num_base_trans

            proj_image = self._project_global(self.base_rot, ctf=ctf)  # shape: (B_or_1, K*Q, P)
            if int(proj_image.shape[0]) == 1:
                proj_image = proj_image.expand(B, *proj_image.shape[1:])  # shape: (B, K*Q, P)

            trans = pose_trans.view(B, 1, 2) + self.base_trans.view(1, T, 2)  # shape: (B, T, 2)
            trans_image = self._translate_global(image, trans)  # shape: (B, T, P)

            hypo_prob = self._evaluate_broadcast(proj_image, trans_image)  # shape: (B*K*Q*T,)
            hypo_dev = hypo_prob.device
            hypo2img_idx = (
                torch.arange(B, device=hypo_dev)
                .view(B, 1, 1, 1)
                .expand(-1, K, Q, T)
                .reshape(-1)
            )  # shape: (B*K*Q*T,)
            hypo2vol_idx = (
                torch.arange(K, device=hypo_dev)
                .view(1, K, 1, 1)
                .expand(B, -1, Q, T)
                .reshape(-1)
            )  # shape: (B*K*Q*T,)
            hypo2rot_idx = (
                torch.arange(Q, device=hypo_dev)
                .view(1, 1, Q, 1)
                .expand(B, K, -1, T)
                .reshape(-1)
            )  # shape: (B*K*Q*T,)
            hypo2trans_idx = (
                torch.arange(T, device=hypo_dev)
                .view(1, 1, 1, T)
                .expand(B, K, Q, -1)
                .reshape(-1)
            )  # shape: (B*K*Q*T,)

            sel_prob, sel2img_idx, sel_payload = self._select_by_prob(
                hypo_prob=hypo_prob,
                hypo2img_idx=hypo2img_idx,
                payload={
                    "vol": hypo2vol_idx,
                    "rot": hypo2rot_idx,
                    "trans": hypo2trans_idx,
                },
            )
            sel2vol_idx = sel_payload["vol"]
            sel2rot_idx = sel_payload["rot"]
            sel2trans_idx = sel_payload["trans"]

            if self.noise is not None:
                proj_flat = proj_image.reshape(B * (K * Q), -1)
                trans_flat = trans_image.reshape(B * T, -1)
                sel2proj_flat_idx = sel2img_idx * (K * Q) + sel2vol_idx * Q + sel2rot_idx
                sel2trans_flat_idx = sel2img_idx * T + sel2trans_idx

            sel_quat = self.base_quat[sel2rot_idx]  # (N_sel, 4)
            sel2rot_grid_idx = so3_grid.get_base_ind(
                sel2rot_idx, self.base_healpix_order
            ).to(self.device, dtype=torch.long)  # (N_sel, 2)

            sel_trans = trans[sel2img_idx, sel2trans_idx]  # (N_sel, 2)
            sel2trans_grid_idx = shift_grid.get_base_ind(
                sel2trans_idx,
                self.trans_grid_samples * 2 ** self.base_trans_healpix_order,
            ).to(self.device, dtype=torch.long)  # (N_sel, 2)

            if sel_quat.ndim == 1:
                sel_quat = sel_quat[None, :]
                sel2rot_grid_idx = sel2rot_grid_idx[None, :]

            if sel_trans.ndim == 1:
                sel_trans = sel_trans[None, :]
                sel2trans_grid_idx = sel2trans_grid_idx[None, :]

            best_quat, best_trans = self._select_best_per_image(
                B=B,
                sel2img_idx=sel2img_idx,
                sel_quat=sel_quat,
                sel_trans=sel_trans,
            )

        else:
            raise ValueError(f"pose_search_scope {self.state.schedule.pose_search_scope} is not supported.")



        # Oversampling progressively increases the HEALPix order and refines the selected
        # candidates. At each round, we subdivide the current candidates to the next grid:
        # - rotations: order o -> o + 1 (8 neighbors)
        # - translations: order t -> t + 1 (4 neighbors)
        #
        # After subdivision, we advance the current rotation / translation orders so that
        # ids are interpreted at the refined grids.
        for oversampling_round in range(self.num_oversampling):
            quat, rot_grid_idx, rotmat = self._subdivide_rot_candidates(
                sel_quat, sel2rot_grid_idx
            )
            rot2img_idx = sel2img_idx.repeat_interleave(8)
            rot2vol_idx = sel2vol_idx.repeat_interleave(8)

            grid_trans, trans_grid_idx = self._subdivide_trans_candidates(
                sel2trans_grid_idx
            )
            trans2img_idx = sel2img_idx.repeat_interleave(4)
            # ``grid_trans`` lives on the refined shared translation grid; add the
            # corresponding per-image pose center to obtain absolute translations.
            trans = pose_trans.index_select(0, trans2img_idx) + grid_trans

            # After subdivision, rot_grid_idx / trans_grid_idx are already at the next
            # rotation / translation orders. Advance both order trackers before computing ids.
            self.current_healpix_order = int(self.base_healpix_order) + oversampling_round + 1
            self.current_trans_healpix_order = int(self.base_trans_healpix_order) + oversampling_round + 1

            # Projection / translation with request de-duplication.
            (
                proj_per_uniq_req,
                unique_proj_req_inverse,
                rot_id,
                unique_proj_req_first_idx,
            ) = self._project_oversampling(
                rot_grid_idx,
                rotmat,
                rot2img_idx=rot2img_idx,
                rot2vol_idx=rot2vol_idx,
                ctf=ctf,
            )
            (
                trans_per_uniq_req,
                unique_trans_req_inverse,
                trans_id,
                unique_trans_req_first_idx,
            ) = self._translate_indexed(
                image,
                trans_grid_idx,
                trans,
                trans2img_idx=trans2img_idx,
            )


            (
                hypo2img_idx,
                hypo2vol_idx,
                hypo2proj_req_idx,
                hypo2trans_req_idx,
            ) = self._unique_hypo_idx_oversampling(
                sel2img_idx=sel2img_idx,
                sel2vol_idx=sel2vol_idx,
                rot_id=rot_id,
                trans_id=trans_id,
                unique_proj_req_inverse=unique_proj_req_inverse,
                unique_trans_req_inverse=unique_trans_req_inverse,
            )

            proj_image = proj_per_uniq_req  # shape: (U_proj, P)
            trans_image = trans_per_uniq_req  # shape: (U_trans, P)

            hypo2proj_idx = hypo2proj_req_idx  # shape: (U_hypo,)
            hypo2trans_idx = hypo2trans_req_idx  # shape: (U_hypo,)

            hypo_prob = self._evaluate_indexed(
                proj_image,
                trans_image,
                B,
                hypo2img_idx,
                hypo2proj_idx,
                hypo2trans_idx,
            )

            # Select candidates 
            sel_prob, sel2img_idx, sel_payload = self._select_by_prob(
                hypo_prob=hypo_prob,
                hypo2img_idx=hypo2img_idx,
                payload={
                    "vol": hypo2vol_idx,
                    "proj_req": hypo2proj_req_idx,
                    "trans_req": hypo2trans_req_idx,
                },
            )

            sel2vol_idx = sel_payload["vol"]
            sel2proj_req_idx = sel_payload["proj_req"]
            sel2trans_req_idx = sel_payload["trans_req"]

            if self.noise is not None:
                proj_flat = proj_image
                trans_flat = trans_image
                sel2proj_flat_idx = sel2proj_req_idx
                sel2trans_flat_idx = sel2trans_req_idx

            sel_quat, sel2rot_grid_idx, sel_trans, sel2trans_grid_idx = (
                self._gather_selected_candidates(
                    quat=quat,
                    rot_grid_idx=rot_grid_idx,
                    trans=trans,
                    trans_grid_idx=trans_grid_idx,
                    unique_proj_req_first_idx=unique_proj_req_first_idx,
                    unique_trans_req_first_idx=unique_trans_req_first_idx,
                    sel2proj_req_idx=sel2proj_req_idx,
                    sel2trans_req_idx=sel2trans_req_idx,
                )
            )

            if sel_quat.ndim == 1:
                sel_quat = sel_quat[None, :]
                sel2rot_grid_idx = sel2rot_grid_idx[None, :]

            if sel_trans.ndim == 1:
                sel_trans = sel_trans[None, :]
                sel2trans_grid_idx = sel2trans_grid_idx[None, :]

            best_quat, best_trans = self._select_best_per_image(
                B=B,
                sel2img_idx=sel2img_idx,
                sel_quat=sel_quat,
                sel_trans=sel_trans,
            )

        self.current_healpix_order = self.base_healpix_order
        self.current_trans_healpix_order = self.base_trans_healpix_order

        self.pose.accumulate(
            particle_index,
            quaternion=best_quat,
            translation=best_trans,
        )
        sel_rotmat = quaternion_to_matrix(sel_quat).to(self.device)

        if self.noise is not None and proj_flat is not None:
            sel_radial_residual_power = self._radial_residual_power(
                proj_flat,
                trans_flat,
                sel2proj_idx=sel2proj_flat_idx,
                sel2trans_idx=sel2trans_flat_idx,
            )

        self.volume_version = getattr(self.volume, "volume_version", self.volume_version)

        return sel_prob, sel2img_idx, sel2vol_idx, sel_rotmat, sel_trans, sel_radial_residual_power