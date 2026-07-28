from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from cryoseed.config import MainConfig
from cryoseed.cryoem.mask import circular_mask
from cryoseed.cryoem.rotation import quaternion_to_matrix, matrix_to_euler
from cryoseed.fft.coords import fftindex_radial2d
from cryoseed.utils.torch_utils import _norm_device
from cryoseed.modules.pose import Pose
from cryoseed.modules.statistics.noise import NoiseVariance
from cryoseed.modules.volume import Volume
from cryoseed.ops.loss import spectral_mse_loss
from cryoseed.ops.radial import radial_residual_power
from cryoseed.ops.transforms import downsample2d, translate_image
from cryoseed.state import OptimState

from . import PoseGeometry, shift_grid, so3_grid


LOGGER = logging.getLogger(__name__)


class EulerPoseSearcher(torch.nn.Module):
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
            neighbor_steps=config.pose_search.neighbor_steps,
            trans_grid_samples=config.pose_search.trans_grid_samples,
            trans_grid_x_shift=config.pose_search.trans_grid_x_shift,
            trans_grid_y_shift=config.pose_search.trans_grid_y_shift,
            pose_chunk_factor=config.pose_search.pose_chunk_factor,
            max_candidates=config.pose_search.max_candidates,
            criterion_chunk=config.pose_search.criterion_chunk,
            candidate_select_threshold=config.pose_search.candidate_select_threshold,
            volume_class_similarity=config.pose_search.volume_class_similarity,
            volume_class_similarity_scope=config.pose_search.volume_class_similarity_scope,
            ring_averaged_mse=config.pose_search.ring_averaged_mse,
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
        neighbor_steps: int = 2,
        trans_grid_samples: int = 5,
        trans_grid_x_shift: int = 0,
        trans_grid_y_shift: int = 0,
        pose_chunk_factor: int = 2560,
        max_candidates: int = -1,
        criterion_chunk: int = 8192,
        candidate_select_threshold: float = 0.999,
        volume_class_similarity: float = 0.0,
        volume_class_similarity_scope: str = "global",
        ring_averaged_mse: bool = False,
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
        self.neighbor_steps = neighbor_steps
        self.trans_grid_samples = trans_grid_samples
        self.trans_grid_x_shift = trans_grid_x_shift
        self.trans_grid_y_shift = trans_grid_y_shift
        self.pose_chunk_factor = pose_chunk_factor
        self.criterion_chunk = criterion_chunk
        self.candidate_select_threshold = candidate_select_threshold
        self.volume_class_similarity = float(volume_class_similarity)
        self.volume_class_similarity_scope = str(volume_class_similarity_scope)
        self.ring_averaged_mse = bool(ring_averaged_mse)

        if self.neighbor_steps < 0:
            raise ValueError("neighbor_steps must be >= 0")
        if self.pose_chunk_factor is not None and self.pose_chunk_factor <= 0:
            raise ValueError("pose_chunk_factor must be positive or None")
        if self.criterion_chunk <= 0:
            raise ValueError("criterion_chunk must be > 0")
        if max_candidates == 0 or max_candidates < -1:
            raise ValueError("max_candidates must be > 0, or -1 for unlimited")
        self.max_candidates = None if max_candidates == -1 else int(max_candidates)
        if not (0 < self.candidate_select_threshold <= 1):
            raise ValueError("candidate_select_threshold must be in (0, 1]")
        if not (0.0 <= self.volume_class_similarity <= 1.0):
            raise ValueError("volume_class_similarity must be in [0, 1]")
        if self.volume_class_similarity_scope not in {"global", "all"}:
            raise ValueError(
                "volume_class_similarity_scope must be one of {'global', 'all'}"
            )

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

        This mirrors the pattern used in :class:`~cryoseed.optim.pose.healpix_searcher.HEALPixPoseSearcher`.

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

        if schedule.side_length is None:
            raise ValueError("schedule.side_length must be set")
        if schedule.healpix_order is None:
            raise ValueError("schedule.healpix_order must be set")
        if schedule.oversampling is None:
            raise ValueError("schedule.oversampling must be set")

        # NOTE: scheduler may store side_length as float (e.g. from heuristic updates).
        # Most downstream ops require an integer side length.
        self.side_length = int(schedule.side_length)
        self.base_healpix_order = int(schedule.healpix_order)
        self.current_healpix_order = int(schedule.healpix_order)
        self.base_trans_healpix_order = 0
        self.current_trans_healpix_order = 0
        self.num_oversampling = int(schedule.oversampling)
        self.trans_grid_extent = float(schedule.trans_grid_extent)
        self.trans_grid_samples = int(
            getattr(schedule, "trans_grid_samples", self.trans_grid_samples)
        )

        if self.side_length <= 0:
            raise ValueError(f"side_length must be > 0, got {self.side_length}")

        if self.base_healpix_order < 0:
            raise ValueError("healpix_order must be >= 0")
        if self.trans_grid_extent < 0:
            raise ValueError("trans_grid_extent must be >= 0")
        if self.trans_grid_samples <= 0:
            raise ValueError("trans_grid_samples must be > 0")

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

    def refresh(self) -> None:
        """Refresh buffers derived from the current schedule (side length, healpix order, etc.)."""
        self._refresh_schedule_state()
        self._refresh_radial_buffers()

    def _expand_current_rot_neighbors(
        self,
        anchor: torch.Tensor,
        *,
        neighbor_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample rotation neighbors around an anchor quaternion.

        This searcher exposes the user-facing name ``neighbor_steps``. The lower-level
        SO(3) helper in ``so3_grid`` still uses the historical argument name
        ``k_steps``, so this method only performs a naming translation at the call site.

        Args:
            anchor: Anchor quaternions of shape ``(N, 4)``.
            neighbor_steps: Neighborhood radius in grid steps. If ``None``, uses
                ``self.neighbor_steps``.

        Returns:
            A tuple ``(quat, rotmat)`` where:

            - quat: Sampled quaternions with shape ``(N, Q, 4)``.
            - rotmat: Rotation matrices with shape ``(N, Q, 3, 3)``.

            Here ``Q = (2 * neighbor_steps + 1) ** 3``.
        """
        current_healpix_order = int(self.current_healpix_order)
        neighbor_steps = int(
            self.neighbor_steps if neighbor_steps is None else neighbor_steps
        )
        Q = int((2 * neighbor_steps + 1) ** 3)

        rotmat_anchor = quaternion_to_matrix(anchor)
        euler_anchor = matrix_to_euler(rotmat_anchor)
        _, quat, rotmat = so3_grid.euler_local_sampling(
            euler_anchor,
            current_healpix_order,
            k_steps=neighbor_steps,
        )

        quat = quat.view(-1, Q, 4).to(self.device)
        rotmat = rotmat.view(-1, Q, 3, 3).to(self.device)

        return quat, rotmat

    def _expand_current_trans_neighbors(
        self,
        anchor: torch.Tensor,
        *,
        neighbor_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample translation neighbors around an anchor translation.

        This searcher exposes the user-facing name ``neighbor_steps``. The lower-level
        translation-grid helper in ``shift_grid`` still uses the historical argument
        name ``k_steps``, so this method only performs a naming translation at the
        call site.

        Args:
            anchor: Anchor translations of shape ``(N, 2)`` in pixels on the input Fourier grid
                (``D = image.shape[-1]`` in :meth:`search`).
            neighbor_steps: Neighborhood radius in grid steps. If ``None``, uses
                ``self.neighbor_steps``.

        Returns:
            Sampled translations with shape ``(N, T, 2)``, where
            ``T = (2 * neighbor_steps + 1) ** 2``.
        """
        trans_healpix_order = int(self.current_trans_healpix_order)
        neighbor_steps = int(
            self.neighbor_steps if neighbor_steps is None else neighbor_steps
        )
        T = int((2 * neighbor_steps + 1) ** 2)

        trans = shift_grid.translation_local_sampling(
            anchor,
            trans_healpix_order,
            self.trans_grid_extent,
            self.trans_grid_samples,
            k_steps=neighbor_steps,
        )
        trans = trans.view(-1, T, 2)

        return trans

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

    def _normalize_anchor_batch(
        self,
        anchor: torch.Tensor,
        *,
        batch_size: int,
        width: int,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(anchor, torch.Tensor):
            anchor = torch.as_tensor(anchor)

        if anchor.ndim == 1:
            if int(anchor.shape[0]) != width:
                raise ValueError(
                    f"{name} must have shape ({width},), (1, {width}), or (B, {width}); "
                    f"got {tuple(anchor.shape)}"
                )
            anchor = anchor.unsqueeze(0)
        elif anchor.ndim != 2 or int(anchor.shape[1]) != width:
            raise ValueError(
                f"{name} must have shape ({width},), (1, {width}), or (B, {width}); "
                f"got {tuple(anchor.shape)}"
            )

        anchor_batch = int(anchor.shape[0])
        if anchor_batch == 1:
            anchor = anchor.expand(batch_size, -1)
        elif anchor_batch != batch_size:
            raise ValueError(
                f"{name} batch must be 1 or match image batch B={batch_size}; "
                f"got {anchor_batch} with shape {tuple(anchor.shape)}"
            )

        return anchor.to(device=self.device, dtype=dtype)

    def _project_local(
        self,
        rotation: torch.Tensor,
        *,
        ctf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project the 3D Fourier volume into 2D Fourier slices (central slices).

        This implementation computes projections on-the-fly. If ``pose_chunk_factor`` is set,
        the computation is chunked to reduce peak memory usage.

        Args:
            rotation: Rotation matrices of shape ``(B, Q, 3, 3)``.
            ctf: Optional CTF tensor of shape ``(B, D, D)``. If provided, the CTF is downsampled
                to ``(B, L, L)``, masked, and applied to the projections.

        Returns:
            Complex Fourier projections of shape ``(B, K * Q, P)``, where ``P`` is the number of
            valid Fourier pixels selected by ``valid_pixel_mask``.
        """
        B = int(rotation.shape[0])
        K = int(self.volume.num_volumes)
        Q = int(rotation.shape[1])
        L = self.side_length
        # Local projection also follows the current autograd context so SGD can
        # use differentiable projections while standard pose search keeps the
        # faster inference/cache path.
        use_grad_proj = torch.is_grad_enabled() and bool(getattr(self.volume, "requires_grad", False))

        rotation = rotation.view(1, B * Q, 3, 3).expand(K, -1, -1, -1)  # (K, B * Q, 3, 3)

        if self.pose_chunk_factor is not None:
            proj_chunk = math.ceil((self.pose_chunk_factor / L) ** 2 / K)
            if use_grad_proj:
                raw_proj = torch.cat(
                    [
                        self.volume(
                            rotation[:, chunk_start:min(chunk_start + proj_chunk, rotation.shape[1])],
                            side_length=L,
                        )
                        for chunk_start in range(0, rotation.shape[1], proj_chunk)
                    ],
                    dim=1,
                )
            else:
                raw_proj = torch.empty(K, B * Q, L, L, dtype=torch.complex64, device=self.device)
                for chunk_start in range(0, rotation.shape[1], proj_chunk):
                    chunk_end = min(chunk_start + proj_chunk, rotation.shape[1])
                    raw_proj[:, chunk_start:chunk_end] = self.volume.project(
                        rotation[:, chunk_start:chunk_end], side_length=L
                    )
        else:
            if use_grad_proj:
                raw_proj = self.volume(rotation, side_length=L)
            else:
                raw_proj = self.volume.project(rotation, side_length=L)  # (K, B * Q, L, L)

        raw_proj = raw_proj.reshape(K * B * Q, L * L)[:, self.valid_pixel_mask]  # (K * B * Q, P)
        if K > 1:
            raw_proj = raw_proj.reshape(K, B, Q, -1).permute(1, 0, 2, 3).contiguous()  # (B, K, Q, P)
            proj = raw_proj.view(B, K * Q, -1)  # (B, K * Q, P)
        else:
            proj = raw_proj.view(B, Q, -1)  # (B, K * Q, P)

        if ctf is not None:
            ctf_valid = self._prepare_ctf_valid_pixels(
                ctf,
                device=proj.device,
                side_length=L,
                dtype=raw_proj.real.dtype,
            )
            ctf_valid = ctf_valid.view(B, 1, -1).expand(-1, K * Q, -1)
            proj = proj * ctf_valid

        return proj

    def project_local(
        self,
        rotation: torch.Tensor,
        *,
        ctf: torch.Tensor | None = None,
        return_geometry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, PoseGeometry]:
        """Project local rotation candidates for the current search state.

        Args:
            rotation: Rotation matrices of shape ``(B, Q, 3, 3)``.
            ctf: Optional CTF tensor of shape ``(B, D, D)`` or ``(B, L, L)``.
            return_geometry: If ``True``, also return the rotation matrices used
                to produce the projections.

        Returns:
            Complex tensor of shape ``(B, K * Q, P)`` over the searcher's valid
            Fourier pixels. When ``return_geometry=True``, returns a tuple
            ``(proj_image, geometry)`` where ``geometry.rotmat`` has shape
            ``(B, Q, 3, 3)``.
        """
        proj_image = self._project_local(rotation, ctf=ctf)
        if return_geometry:
            return proj_image, PoseGeometry(rotmat=rotation)
        return proj_image

    def _project_oversampling(
        self,
        rotation: torch.Tensor,
        *,
        sel2vol_idx: torch.LongTensor,
        ctf: torch.Tensor | None = None,
        sel2img_idx: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Project the 3D Fourier volume into 2D Fourier slices (central slices).

        This is the oversampling/refinement variant where each selected hypothesis already
        carries a fixed volume index via ``sel2vol_idx``.

        Args:
            rotation: Rotation matrices of shape ``(N, Q, 3, 3)``.
            sel2vol_idx: Volume indices for each selected hypothesis, shape ``(N,)``.
            ctf: Optional CTF tensor of shape ``(B, D, D)``. If provided, the CTF is downsampled
                to ``(B, L, L)``, masked, and applied to the projections.
            sel2img_idx: Image indices for each selected hypothesis, shape ``(N,)``. Required
                when applying CTF.

        Returns:
            Complex Fourier projections of shape ``(N, Q, P)``, where ``P`` is the number of
            valid Fourier pixels selected by ``valid_pixel_mask``.
        """
        N = int(rotation.shape[0])
        K = int(self.volume.num_volumes)
        Q = int(rotation.shape[1])
        L = self.side_length

        rotation = rotation.view(1, N * Q, 3, 3).expand(K, -1, -1, -1)  # (K, N * Q, 3, 3)

        if self.pose_chunk_factor is not None:
            proj_chunk = math.ceil((self.pose_chunk_factor / L) ** 2 / K)
            raw_proj = torch.empty(K, N * Q, L, L, dtype=torch.complex64, device=self.device)
            for chunk_start in range(0, rotation.shape[1], proj_chunk):
                chunk_end = min(chunk_start + proj_chunk, rotation.shape[1])
                raw_proj[:, chunk_start:chunk_end] = self.volume.project(
                    rotation[:, chunk_start:chunk_end], side_length=L
                )
        else:
            raw_proj = self.volume.project(rotation, side_length=L)  # (K, N * Q, L, L)

        raw_proj = raw_proj.reshape(K * N * Q, L * L)[:, self.valid_pixel_mask]  # (K * N * Q, P)
        if K > 1:
            raw_proj = raw_proj.reshape(K, N, Q, -1)  # (K, N, Q, P)
            sel_idx = torch.arange(N, device=sel2vol_idx.device)
            proj = raw_proj[sel2vol_idx, sel_idx]  # (N, Q, P)
        else:
            proj = raw_proj.view(N, Q, -1)  # (N, Q, P)

        if ctf is not None:
            ctf_valid = self._prepare_ctf_valid_pixels(
                ctf,
                device=proj.device,
                side_length=L,
                dtype=raw_proj.real.dtype,
            )
            ctf_valid = ctf_valid[sel2img_idx].view(N, 1, -1).expand(-1, Q, -1)
            proj = proj * ctf_valid

        return proj
    
    def _translate(
        self,
        image: torch.Tensor,
        translation: torch.Tensor,
        *,
        img_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Apply 2D translations to Fourier-domain images.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex). Translations are
                applied on this full-resolution FFT grid and the result is then center-cropped
                to ``(B, L, L)`` using ``side_length``.
            translation: Translation vectors of shape ``(N, T, 2)`` in pixels on the input grid
                ``D`` (not scaled by ``side_length``), where each row is ``(dx, dy)``.

        Returns:
            Complex tensor of translated images with shape ``(N, T, P)``, where ``P`` is the
            number of valid Fourier pixels selected by ``valid_pixel_mask``.
        """
        N = int(translation.shape[0])
        T = int(translation.shape[1])
        D = int(image.shape[-1])
        L = int(self.side_length)

        device = self.device
        image = image.index_select(0, img_idx.to(device))
        translation = translation.to(device=device)

        # NOTE:
        # - translate_image interprets (dx, dy) in pixel units of the FFT grid of its input.
        # - We translate on the original D×D grid (for correct units) and then crop to side_length L.
        if self.pose_chunk_factor is not None:
            trans_chunk = math.ceil((self.pose_chunk_factor / D) ** 2)
            trans_image = torch.empty(N, T, D, D, dtype=image.dtype, device=image.device)
            for chunk_start in range(0, T, trans_chunk):
                chunk_end = min(chunk_start + trans_chunk, T)
                trans = translation[:, chunk_start:chunk_end, :]
                trans_image[:, chunk_start:chunk_end] = translate_image(image, trans)
        else:
            trans_image = translate_image(image, translation)  # (N, T, D, D)
            
        trans_image = downsample2d(trans_image, L)  # (N, T, L, L)

        mask = self.valid_pixel_mask
        if mask.device != trans_image.device:
            mask = mask.to(trans_image.device)
        return trans_image.reshape(N, T, L * L)[:, :, mask].contiguous()  # (N, T, P)

    def translate_local(
        self,
        image: torch.Tensor,
        translation: torch.Tensor,
        *,
        geometry: PoseGeometry | None = None,
        return_geometry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, PoseGeometry]:
        """Translate local candidates with the default per-image pairing.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)``.
            translation: Local translation candidates of shape ``(B, T, 2)`` in
                input FFT-grid pixel units.
            geometry: Optional geometry container to augment with the active
                translation candidates.
            return_geometry: If ``True``, also return the translations used to
                produce the translated images.

        Returns:
            Complex tensor of shape ``(B, T, P)`` over the searcher's valid
            Fourier pixels. When ``return_geometry=True``, returns a tuple
            ``(trans_image, geometry)`` where ``geometry.trans`` has shape
            ``(B, T, 2)``.
        """
        B = int(image.shape[0])
        if int(translation.shape[0]) != B:
            raise ValueError(
                f"translation batch must match image batch for local translation: "
                f"expected B={B}, got translation.shape[0]={int(translation.shape[0])}"
            )
        img_idx = torch.arange(B, device=self.device, dtype=torch.long)
        trans_image = self._translate(image, translation, img_idx=img_idx)
        if return_geometry:
            if geometry is None:
                geometry = PoseGeometry()
            return trans_image, geometry.merged(trans=translation)
        return trans_image

    def _evaluate(
        self,
        proj_image: torch.Tensor,
        trans_image: torch.Tensor,
        num_images: int,
        hypo2img_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute per-image probabilities over a Cartesian hypothesis grid.

        This routine is used by both:

        - the initial local search, where ``proj_image`` enumerates ``K * Q`` (volume x rotation)
          candidates per anchor image; and
        - oversampling refinement, where the volume index is already fixed per anchor and
          ``proj_image`` typically enumerates only ``Q`` rotation candidates.

        Let ``N = proj_image.shape[0]`` be the number of anchors evaluated in this call
        (``N == B`` in the initial local search, and ``N == N_sel`` during oversampling),
        and let ``B = num_images`` be the number of distinct images whose probabilities are
        normalized independently. Hypotheses are grouped by image via ``hypo2img_idx``.

        Numerical stability:
            Probabilities are computed as ``exp(-mse)``. To reduce underflow, the per-image
            minimum MSE is subtracted before exponentiation.
 
        Distributed evaluation:
            If ``torch.distributed`` is initialized and ``device_mesh`` is provided, the MSE
            computation is sharded across the projection-candidate dimension ``M`` using
            ``device_mesh.get_group(1)``. Each rank writes its slice into the output tensor and
            an ``all_reduce`` sum assembles the full MSE on every rank.

        Args:
            proj_image: Complex projections with shape ``(N, M, P)``, where ``M`` is the number of
                projection candidates per anchor (e.g. ``M = K * Q`` for local search and
                ``M = Q`` for oversampling).
            trans_image: Complex translated images with shape ``(N, T, P)``, where ``T`` is the
                number of translation candidates per anchor.
            num_images: Number of distinct images ``B`` in the current batch. Used for per-image
                normalization; all entries of ``hypo2img_idx`` must be in ``[0, B-1]``.
            hypo2img_idx: Image index for each hypothesis in the implicit ``(N, M, T)`` grid,
                flattened to shape ``(N * M * T,)``.

        Returns:
            A 1D tensor of probabilities with shape ``(N * M * T,)``, normalized independently
            for each image id in ``hypo2img_idx``.
        """
        B = int(num_images)
        N = int(proj_image.shape[0])
        KQ = int(proj_image.shape[1])
        T = int(trans_image.shape[1])
 
        if int(trans_image.shape[0]) != N:
            raise ValueError(
                f"trans_image batch dim must match proj_image: {int(trans_image.shape[0])} vs {N}"
            )


        @torch.no_grad()
        def compute_mse_broadcast(
            proj_image: torch.Tensor,
            trans_image: torch.Tensor,
        ) -> torch.Tensor:
            device = self.device

            R = int(self.R)
            if self.noise is None:
                precision = torch.ones((R,), device=device, dtype=torch.float32)
            else:
                precision = self.noise.precision[:R].to(device)
            finite = torch.isfinite(precision)

            weight_r = torch.zeros_like(precision)
            weight_r[finite] = 0.5 * precision[finite]
            weight_r[0] = 0
            if self.ring_averaged_mse:
                weight_r = weight_r * self.ring_denom.to(device)

            weight = weight_r[self.valid_pixel2ring_idx.to(device)].contiguous()  # (P,)


            if proj_image.dim() != 3 or trans_image.dim() != 3:
                raise ValueError(
                    f"expected proj_image/trans_image to be flattened (B,*,P); got {tuple(proj_image.shape)} and {tuple(trans_image.shape)}"
                )

            mse = torch.zeros((N, KQ, T), device=device, dtype=torch.float32)

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
                criterion_chunk = self.criterion_chunk
                for chunk_start in range(
                    slice_start, slice_end, criterion_chunk
                ):
                    chunk_end = min(chunk_start + criterion_chunk, slice_end)
                    spectral_mse_loss(
                        proj_image[:, chunk_start:chunk_end],
                        trans_image,
                        weight=weight,
                        out=mse[:, chunk_start:chunk_end, :],
                        spectral_reduction="sum",
                        reduction="none",
                    )

            if (
                dist is not None
                and dist.is_available()
                and dist.is_initialized()
                and calculation_parallel_size > 1
            ):
                dist.all_reduce(mse, dist.ReduceOp.SUM, group)

            return mse.view(-1)

        mse = compute_mse_broadcast(proj_image, trans_image)  # (N * KQ * T)

        # Tackle underflow in exp()
        min_per_img = torch.full((B,), float("inf"), device=mse.device)  # (B,)
        min_per_img = min_per_img.scatter_reduce(0, hypo2img_idx, mse, reduce="amin")

        mse = mse - min_per_img[hypo2img_idx]
        prob = torch.exp(-mse)  # (N * KQ * T)
        sum_per_img = torch.zeros((B,), device=prob.device).scatter_add(0, hypo2img_idx, prob)
        prob = prob / sum_per_img[hypo2img_idx]
        prob = prob.reshape(-1)  # (N * KQ * T,)

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
                    local: {"vol": hypo2vol_idx, "proj_req": hypo2proj_req_idx, "trans_req": hypo2trans_req_idx}

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

    def _apply_volume_class_similarity(
        self,
        *,
        hypo_prob: torch.Tensor,
        hypo2img_idx: torch.LongTensor,
        hypo2vol_idx: torch.LongTensor,
        num_images: int,
        num_volumes: int,
    ) -> torch.Tensor:
        similarity = float(self.volume_class_similarity)
        if num_volumes <= 1 or similarity <= 0.0:
            return hypo_prob
        if (
            self.volume_class_similarity_scope == "global"
            and self.state.schedule.pose_search_scope != "global"
        ):
            return hypo_prob

        flat_class_idx = hypo2img_idx * int(num_volumes) + hypo2vol_idx
        marginal = torch.zeros(
            int(num_images) * int(num_volumes),
            device=hypo_prob.device,
            dtype=hypo_prob.dtype,
        )
        marginal = marginal.scatter_add(0, flat_class_idx, hypo_prob).view(
            int(num_images), int(num_volumes)
        )
        mixed_marginal = (1.0 - similarity) * marginal + similarity / float(num_volumes)

        eps = torch.finfo(hypo_prob.dtype).tiny
        scale = mixed_marginal.reshape(-1)[flat_class_idx] / marginal.reshape(-1)[
            flat_class_idx
        ].clamp_min(eps)
        adjusted_prob = hypo_prob * scale
        sum_per_img = torch.zeros(
            int(num_images),
            device=adjusted_prob.device,
            dtype=adjusted_prob.dtype,
        ).scatter_add(0, hypo2img_idx, adjusted_prob)
        return adjusted_prob / sum_per_img[hypo2img_idx].clamp_min(eps)

    def _select_best_per_image(
        self,
        *,
        B: int,
        sel_prob: torch.Tensor,
        sel2img_idx: torch.LongTensor,
        sel2vol_idx: torch.LongTensor,
        sel_quat: torch.Tensor,
        sel_trans: torch.Tensor,
    ):
        # Assumes selected candidates are grouped by image index, and within each image
        # group the first candidate is already the highest-probability one.
        best_vol_idx = torch.zeros((B,), device=self.device, dtype=torch.long)
        best_quat = torch.zeros((B, 4), device=self.device, dtype=torch.float32)
        best_trans = torch.zeros((B, 2), device=self.device, dtype=torch.float32)
        best_confidence = torch.zeros((B,), device=self.device, dtype=sel_prob.dtype)
        best_volume_class_confidence = torch.zeros((B,), device=self.device, dtype=sel_prob.dtype)

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

        best_vol_idx.index_copy_(0, sel_img_idx_unique, sel2vol_idx[sel_best_src_idx])
        best_quat.index_copy_(0, sel_img_idx_unique, sel_quat[sel_best_src_idx])
        best_trans.index_copy_(0, sel_img_idx_unique, sel_trans[sel_best_src_idx])
        best_confidence.index_copy_(0, sel_img_idx_unique, sel_prob[sel_best_src_idx])

        group_start = 0
        num_volumes = int(self.volume.num_volumes)
        for img_idx, group_size, best_src_idx in zip(
            sel_img_idx_unique.tolist(),
            sel_img_counts.tolist(),
            sel_best_src_idx.tolist(),
        ):
            group_end = group_start + group_size
            volume_prob = torch.zeros(
                (num_volumes,), device=self.device, dtype=sel_prob.dtype
            )
            volume_prob.scatter_add_(
                0,
                sel2vol_idx[group_start:group_end],
                sel_prob[group_start:group_end],
            )
            best_volume_class_confidence[img_idx] = volume_prob[sel2vol_idx[best_src_idx]]
            group_start = group_end

        return (
            best_vol_idx,
            best_quat,
            best_trans,
            best_confidence,
            best_volume_class_confidence,
        )

    @torch.no_grad()
    def _search_from_anchor(
        self,
        image: torch.Tensor,
        *,
        quaternion: torch.Tensor,
        translation: torch.Tensor,
        ctf: torch.Tensor | None = None,
        particle_index: torch.LongTensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        criterion = self.state.schedule.pose_search_criterion
        if criterion != "posterior":
            raise ValueError(
                "Euler pose search only supports pose_search_criterion='posterior'"
            )
        device = self.device

        image = image.to(device)
        B = int(image.shape[0])
        K = int(self.volume.num_volumes)

        quaternion = self._normalize_anchor_batch(
            quaternion,
            batch_size=B,
            width=4,
            name="quaternion",
            dtype=torch.float32,
        )
        quaternion = F.normalize(quaternion, dim=-1)
        translation = self._normalize_anchor_batch(
            translation,
            batch_size=B,
            width=2,
            name="translation",
            dtype=self.pose.trans.dtype,
        )

        if particle_index is not None:
            particle_index = particle_index.to(device=self.pose.device, dtype=torch.long)
            if int(particle_index.shape[0]) != B:
                raise ValueError(
                    f"particle_index batch must match image batch: expected B={B}, "
                    f"got {int(particle_index.shape[0])}"
                )

        # NOTE:
        # - Translations are always expressed in pixels of the input FFT grid (D = image.shape[-1]).
        # - Even though the posterior criterion uses a cropped side_length-L window, _translate applies shifts on D×D
        #   first and only then center-crops to L.

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

        if self.state.schedule.pose_search_scope == "local":
            neighbor_steps = int(self.neighbor_steps)

            img_idx = torch.arange(B, device=device, dtype=torch.long)  # shape: (B,)
            quat = quaternion
            trans = translation

            quat, rotmat = self._expand_current_rot_neighbors(
                quat, neighbor_steps=neighbor_steps
            ) # (B, Q, 4)
            trans = self._expand_current_trans_neighbors(
                trans, neighbor_steps=neighbor_steps
            ) # (B, T, 2)

            Q = int(quat.shape[1])
            T = int(trans.shape[1])

            proj_image = self._project_local(rotmat, ctf=ctf) # (B, K * Q, P)
            trans_image = self._translate(image, trans, img_idx=img_idx) # (B, T, P)

            hypo2img_idx = img_idx.view(B, 1, 1, 1).expand(-1, K, Q, T).reshape(-1)
            hypo2vol_idx = (
                torch.arange(K, device=hypo2img_idx.device, dtype=torch.long)
                .view(1, K, 1, 1)
                .expand(B, -1, Q, T)
                .reshape(-1)
            )
            hypo2rot_idx = (
                torch.arange(B * Q, device=hypo2img_idx.device, dtype=torch.long)
                .view(B, 1, Q, 1)
                .expand(-1, K, -1, T)
                .reshape(-1)
            )
            hypo2trans_idx = (
                torch.arange(B * T, device=hypo2img_idx.device, dtype=torch.long)
                .view(B, 1, 1, T)
                .expand(-1, K, Q, -1)
                .reshape(-1)
            )
            hypo_prob = self._evaluate(proj_image, trans_image, B, hypo2img_idx)
            hypo_prob = self._apply_volume_class_similarity(
                hypo_prob=hypo_prob,
                hypo2img_idx=hypo2img_idx,
                hypo2vol_idx=hypo2vol_idx,
                num_images=B,
                num_volumes=K,
            )

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
                local_rot_idx = sel2rot_idx % Q
                local_trans_idx = sel2trans_idx % T
                sel2proj_flat_idx = sel2img_idx * (K * Q) + sel2vol_idx * Q + local_rot_idx
                sel2trans_flat_idx = sel2img_idx * T + local_trans_idx

            sel_quat = quat.view(B * Q, -1)[sel2rot_idx]
            sel_trans = trans.view(B * T, -1)[sel2trans_idx]

            (
                best_vol_idx,
                best_quat,
                best_trans,
                best_confidence,
                best_volume_class_confidence,
            ) = self._select_best_per_image(
                B=B,
                sel_prob=sel_prob,
                sel2img_idx=sel2img_idx,
                sel2vol_idx=sel2vol_idx,
                sel_quat=sel_quat,
                sel_trans=sel_trans,
            )

        else:
            raise ValueError(f"pose_search_scope {self.state.schedule.pose_search_scope} is not supported.")


        for oversampling_round in range(self.num_oversampling):
            self.current_healpix_order = int(self.base_healpix_order) + oversampling_round + 1
            self.current_trans_healpix_order = int(self.base_trans_healpix_order) + oversampling_round + 1
            neighbor_steps = 1

            quat, rotmat = self._expand_current_rot_neighbors(
                sel_quat, neighbor_steps=neighbor_steps
            )  # (N, Q, 4)
            trans = self._expand_current_trans_neighbors(
                sel_trans, neighbor_steps=neighbor_steps
            )  # (N, T, 2)

            N = int(sel2img_idx.shape[0])
            Q = int(quat.shape[1])
            T = int(trans.shape[1])
            
            proj_image = self._project_oversampling(
                rotmat, sel2vol_idx=sel2vol_idx, ctf=ctf, sel2img_idx=sel2img_idx
            ) # (N, Q, P)
            trans_image = self._translate(image, trans, img_idx=sel2img_idx) # (N, T, P)

            hypo2img_idx = sel2img_idx.view(N, 1, 1).expand(-1, Q, T).reshape(-1)
            hypo2vol_idx = sel2vol_idx.view(N, 1, 1).expand(-1, Q, T).reshape(-1)
            hypo2rot_idx = (
                torch.arange(N * Q, device=hypo2img_idx.device, dtype=torch.long)
                .view(N, Q, 1)
                .expand(-1, -1, T)
                .reshape(-1)
            )
            hypo2trans_idx = (
                torch.arange(N * T, device=hypo2img_idx.device, dtype=torch.long)
                .view(N, 1, T)
                .expand(-1, Q, -1)
                .reshape(-1)
            )

            hypo_prob = self._evaluate(proj_image, trans_image, B, hypo2img_idx)

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
                proj_flat = proj_image.view(N * Q, -1)
                trans_flat = trans_image.view(N * T, -1)
                sel2proj_flat_idx = sel2rot_idx
                sel2trans_flat_idx = sel2trans_idx

            sel_quat = quat.view(N * Q, -1)[sel2rot_idx]
            sel_trans = trans.view(N * T, -1)[sel2trans_idx]

            (
                best_vol_idx,
                best_quat,
                best_trans,
                best_confidence,
                best_volume_class_confidence,
            ) = self._select_best_per_image(
                B=B,
                sel_prob=sel_prob,
                sel2img_idx=sel2img_idx,
                sel2vol_idx=sel2vol_idx,
                sel_quat=sel_quat,
                sel_trans=sel_trans,
            )

        self.current_healpix_order = self.base_healpix_order
        self.current_trans_healpix_order = self.base_trans_healpix_order

        if particle_index is not None:
            self.pose.accumulate(
                particle_index,
                quaternion=best_quat,
                translation=best_trans,
                volume_index=best_vol_idx,
                confidence=best_confidence,
                volume_class_confidence=best_volume_class_confidence,
            )

        sel_rotmat = quaternion_to_matrix(sel_quat).to(self.device)

        if (
            self.noise is not None
            and proj_flat is not None
            and not self.state.schedule.full_backprojection
        ):
            sel_radial_residual_power = radial_residual_power(
                proj_flat,
                trans_flat,
                input_indices=sel2proj_flat_idx,
                target_indices=sel2trans_flat_idx,
                side_length=int(self.side_length),
                max_radius=int(self.R) - 1,
                ndim=2,
                use_cache=True,
            )

        return sel_prob, sel2img_idx, sel2vol_idx, sel_rotmat, sel_trans, sel_radial_residual_power

    @torch.no_grad()
    def search_from_anchor(
        self,
        image: torch.Tensor,
        *,
        quaternion: torch.Tensor,
        translation: torch.Tensor,
        ctf: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Run local pose search from explicit quaternion / translation anchors.

        This is the local-search counterpart to the standard
        ``search(image, particle_index=...)`` path. Instead of reading anchor
        poses from ``self.pose`` via dataset-bound particle indices, callers
        provide explicit quaternion / translation anchors directly.

        Batch broadcast is supported:

        - ``quaternion`` may have shape ``(4,)``, ``(1, 4)``, or ``(B, 4)``
        - ``translation`` may have shape ``(2,)``, ``(1, 2)``, or ``(B, 2)``

        The normalized anchors are then expanded into the usual local
        rotation/translation neighborhoods before evaluating posterior
        candidates, so the downstream local-search behavior stays aligned with
        the regular Euler search path.
        """
        return self._search_from_anchor(
            image,
            quaternion=quaternion,
            translation=translation,
            ctf=ctf,
        )

    @torch.no_grad()
    def search_no_grad(
        self,
        image: torch.Tensor,
        *,
        particle_index: torch.LongTensor,
        ctf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
        """Run local pose search (plus oversampling refinement).

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex).
            particle_index: Particle indices of shape ``(B,)``. Used as anchors for local search.
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
              noise estimation is enabled and full backprojection is disabled.

              Unit convention: translations are expressed in pixels of the *input* Fourier grid
              (``D = image.shape[-1]``) and are not scaled by the current ``side_length``.
        """
        if particle_index is None:
            raise ValueError("particle_index is required for Euler pose search")
        particle_index = particle_index.to(device=self.pose.device, dtype=torch.long)
        quaternion = self.pose.quaternion(particle_index)
        translation = self.pose.translation(particle_index).detach().to(
            device=self.device,
            dtype=self.pose.trans.dtype,
        )
        return self._search_from_anchor(
            image,
            quaternion=quaternion,
            translation=translation,
            ctf=ctf,
            particle_index=particle_index,
        )

    def search_grad(
        self,
        image: torch.Tensor,
        *,
        particle_index: torch.LongTensor,
        ctf: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Run the differentiable local Euler pose-search route.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex).
            particle_index: Particle indices of shape ``(B,)``. Used as anchors for local search.
            ctf: Optional per-image CTF tensor of shape ``(B, D, D)`` (or ``(B, L, L)``).
                The batch dimension must match ``image.shape[0]`` (no broadcasting).

        Returns:
            A 7-tuple ``(loss, sel_prob, sel2img_idx, sel2vol_idx, sel_rotmat, sel_trans,
            sel_radial_residual_power)``, where:

            - loss (torch.Tensor): Differentiable data term averaged over the batch.
            - sel_prob (torch.Tensor): Selected hypothesis weights with shape ``(N_sel,)``.
              Candidates are grouped by image index (``sel2img_idx``) and re-normalized within
              each image after selection.
            - sel2img_idx (torch.LongTensor): Image indices for each selected hypothesis with shape
              ``(N_sel,)``.
            - sel2vol_idx (torch.LongTensor): Volume/class indices for each selected hypothesis with
              shape ``(N_sel,)``.
            - sel_rotmat (torch.Tensor): Rotation matrices for each selected hypothesis with shape
              ``(N_sel, 3, 3)``.
            - sel_trans (torch.Tensor): 2D translations (in pixels) for each selected hypothesis
              with shape ``(N_sel, 2)``.
            - sel_radial_residual_power (torch.Tensor | None): Optional per-hypothesis radial
              residual power with shape ``(N_sel, side_length // 2 + 1)``. Returned only when
              noise estimation is enabled and full backprojection is disabled.
        """
        if self.state.schedule.pose_search_scope != "local":
            raise ValueError(
                f"pose_search_scope {self.state.schedule.pose_search_scope} is not supported."
            )
        criterion = self.state.schedule.pose_search_criterion
        if criterion != "posterior":
            raise ValueError(
                "Euler pose search only supports pose_search_criterion='posterior'"
            )
        if int(self.num_oversampling) != 0:
            raise ValueError("Euler differentiable pose search does not support oversampling > 0")
        if particle_index is None:
            raise ValueError("particle_index is required for Euler pose search")
        if self.pose is None:
            raise ValueError("pose is required for Euler pose search")

        device = self.device
        image = image.to(device)
        B = int(image.shape[0])
        K = int(self.volume.num_volumes)

        particle_index = particle_index.to(device=self.pose.device, dtype=torch.long)
        quaternion = self.pose.quaternion(particle_index)
        translation = self.pose.translation(particle_index).detach().to(
            device=device,
            dtype=self.pose.trans.dtype,
        )

        quaternion = self._normalize_anchor_batch(
            quaternion,
            batch_size=B,
            width=4,
            name="quaternion",
            dtype=torch.float32,
        )
        quaternion = F.normalize(quaternion, dim=-1)
        translation = self._normalize_anchor_batch(
            translation,
            batch_size=B,
            width=2,
            name="translation",
            dtype=self.pose.trans.dtype,
        )

        if int(particle_index.shape[0]) != B:
            raise ValueError(
                f"particle_index batch must match image batch: expected B={B}, "
                f"got {int(particle_index.shape[0])}"
            )

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

        with torch.enable_grad():
            neighbor_steps = int(self.neighbor_steps)
            img_idx = torch.arange(B, device=device, dtype=torch.long)

            quat, rotmat = self._expand_current_rot_neighbors(
                quaternion, neighbor_steps=neighbor_steps
            )  # (B, Q, 4), (B, Q, 3, 3)
            trans = self._expand_current_trans_neighbors(
                translation, neighbor_steps=neighbor_steps
            )  # (B, T, 2)

            Q = int(quat.shape[1])
            T = int(trans.shape[1])

            proj_image = self._project_local(rotmat, ctf=ctf)  # (B, K * Q, P)
            trans_image = self._translate(image, trans, img_idx=img_idx)  # (B, T, P)

            R = int(self.R)
            if self.noise is None:
                precision = torch.ones((R,), device=device, dtype=torch.float32)
            else:
                precision = self.noise.precision[:R].to(device=device, dtype=torch.float32)
            finite = torch.isfinite(precision)
            weight_r = torch.zeros_like(precision)
            weight_r[finite] = 0.5 * precision[finite]
            weight_r[0] = 0
            if self.ring_averaged_mse:
                weight_r = weight_r * self.ring_denom.to(device=device, dtype=weight_r.dtype)
            weight = weight_r[self.valid_pixel2ring_idx.to(device=device)].contiguous()

            mse = spectral_mse_loss(
                proj_image,
                trans_image,
                weight=weight,
                reduction="none",
                spectral_reduction="sum",
            ).view(B, K, Q, T)
            mse_flat = mse.reshape(B, K * Q * T)
            nll_per_image = math.log(K * Q * T) - torch.logsumexp(-mse_flat, dim=-1)
            loss = nll_per_image.mean()

            hypo_prob = torch.softmax(-mse_flat, dim=-1).reshape(-1)
            hypo_dev = hypo_prob.device
            hypo2img_idx = (
                torch.arange(B, device=hypo_dev)
                .view(B, 1, 1, 1)
                .expand(-1, K, Q, T)
                .reshape(-1)
            )
            hypo2vol_idx = (
                torch.arange(K, device=hypo_dev)
                .view(1, K, 1, 1)
                .expand(B, -1, Q, T)
                .reshape(-1)
            )
            hypo2rot_idx = (
                torch.arange(B * Q, device=hypo_dev, dtype=torch.long)
                .view(B, 1, Q, 1)
                .expand(-1, K, -1, T)
                .reshape(-1)
            )
            hypo2trans_idx = (
                torch.arange(B * T, device=hypo_dev, dtype=torch.long)
                .view(B, 1, 1, T)
                .expand(-1, K, Q, -1)
                .reshape(-1)
            )
            hypo_prob = self._apply_volume_class_similarity(
                hypo_prob=hypo_prob,
                hypo2img_idx=hypo2img_idx,
                hypo2vol_idx=hypo2vol_idx,
                num_images=B,
                num_volumes=K,
            )

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

            sel_quat = quat.view(B * Q, -1)[sel2rot_idx]
            sel_trans = trans.view(B * T, -1)[sel2trans_idx]
            sel_rotmat = quaternion_to_matrix(sel_quat).to(self.device)

        (
            best_vol_idx,
            best_quat,
            best_trans,
            best_confidence,
            best_volume_class_confidence,
        ) = self._select_best_per_image(
            B=B,
            sel_prob=sel_prob,
            sel2img_idx=sel2img_idx,
            sel2vol_idx=sel2vol_idx,
            sel_quat=sel_quat,
            sel_trans=sel_trans,
        )

        self.current_healpix_order = self.base_healpix_order
        self.current_trans_healpix_order = self.base_trans_healpix_order

        with torch.no_grad():
            self.pose.accumulate(
                particle_index,
                quaternion=best_quat,
                translation=best_trans,
                volume_index=best_vol_idx,
                confidence=best_confidence,
                volume_class_confidence=best_volume_class_confidence,
            )

        if self.noise is not None and not self.state.schedule.full_backprojection:
            with torch.no_grad():
                proj_flat = proj_image.reshape(B * (K * Q), -1)
                trans_flat = trans_image.reshape(B * T, -1)
                local_rot_idx = sel2rot_idx % Q
                local_trans_idx = sel2trans_idx % T
                sel2proj_flat_idx = sel2img_idx * (K * Q) + sel2vol_idx * Q + local_rot_idx
                sel2trans_flat_idx = sel2img_idx * T + local_trans_idx
                sel_radial_residual_power = radial_residual_power(
                    proj_flat,
                    trans_flat,
                    input_indices=sel2proj_flat_idx,
                    target_indices=sel2trans_flat_idx,
                    side_length=int(self.side_length),
                    max_radius=int(self.R) - 1,
                    ndim=2,
                    use_cache=True,
                )

        return (
            loss,
            sel_prob,
            sel2img_idx,
            sel2vol_idx,
            sel_rotmat,
            sel_trans,
            sel_radial_residual_power,
        )

    def search(
        self,
        image: torch.Tensor,
        *,
        particle_index: torch.LongTensor,
        ctf: torch.Tensor | None = None,
        mode: str = "auto",
    ):
        """Dispatch to the gradient-enabled or no-grad Euler search route.

        Args:
            image: Fourier-domain images of shape ``(B, D, D)`` (complex).
            particle_index: Particle indices of shape ``(B,)``. Used as anchors for local search.
            ctf: Optional per-image CTF tensor of shape ``(B, D, D)`` (or ``(B, L, L)``).
                The batch dimension must match ``image.shape[0]`` (no broadcasting).
            mode: Search execution mode. ``"grad"`` dispatches to :meth:`search_grad`,
                ``"no_grad"`` dispatches to :meth:`search_no_grad`, and ``"auto"``
                dispatches to :meth:`search_grad` when autograd is enabled and the
                volume requires gradients.

        Returns:
            The return value of the selected search route.
        """
        if mode == "grad":
            return self.search_grad(
                image,
                particle_index=particle_index,
                ctf=ctf,
            )
        if mode == "auto":
            if torch.is_grad_enabled() and bool(getattr(self.volume, "requires_grad", False)):
                return self.search_grad(
                    image,
                    particle_index=particle_index,
                    ctf=ctf,
                )
            return self.search_no_grad(
                image,
                particle_index=particle_index,
                ctf=ctf,
            )
        if mode == "no_grad":
            return self.search_no_grad(
                image,
                particle_index=particle_index,
                ctf=ctf,
            )
        raise ValueError(f"Unsupported search mode: {mode!r}")