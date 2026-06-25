"""SO(3) grid utilities based on the Hopf-fibration construction.

This implementation follows the incremental SO(3) grid parameterization from
Yershova et al., "Generating uniform incremental grids on SO(3) using the
Hopf fibration."

Compared with the reference construction, this file also includes several
engineering-oriented adaptations used by the search code:

1. Cached HEALPix angle tables to avoid repeated ``healpy.pix2ang`` calls for
   standard resolutions.
2. Tensor-friendly wrappers so the main search path can stay on the active
   PyTorch device as long as possible.
3. Batched child expansion and top-k neighbor selection for refinement steps.
4. A small amount of legacy-compatible NumPy functionality kept for older
   helper paths.
"""

import json
import os

from typing import Tuple, Union

import numpy as np
import torch

from cryoseed.cryoem.rotation import quaternion_to_matrix, matrix_to_quaternion, matrix_to_euler, euler_to_matrix


def grid_s1(resol: int) -> np.ndarray:
    """Return the S1 grid angles for a resolution level.

    Args:
        resol: Resolution level of the SO(3) grid.

    Returns:
        Array of shape ``(6 * 2**resol,)`` containing the S1 fiber angles.
    """
    Npix = 6 * 2**resol
    dt = 2 * np.pi / Npix
    grid = np.arange(Npix) * dt + dt / 2
    return grid


def grid_s2(resol: int):
    """Return the HEALPix S2 grid angles for a resolution level.

    Args:
        resol: Resolution level of the SO(3) grid.

    Returns:
        Tuple ``(theta, phi)`` of arrays with shape ``(12 * 4**resol,)``.
    """
    Nside = 2**resol
    Npix = 12 * Nside * Nside
    theta, phi = pix2ang(Nside, np.arange(Npix), nest=True)
    return theta, phi


def hopf_to_quat(theta, phi, psi) -> np.ndarray:
    """Convert Hopf coordinates to quaternions.

    Args:
        theta: Polar angles in ``[0, pi)``.
        phi: Azimuthal angles in ``[0, 2 * pi)``.
        psi: Fiber angles in ``[0, 2 * pi)``.

    Returns:
        Array of shape ``(..., 4)`` with scalar-first quaternions.
    """
    ct = np.cos(theta / 2)
    st = np.sin(theta / 2)
    quat = np.array(
        [
            ct * np.cos(psi / 2),
            ct * np.sin(psi / 2),
            st * np.cos(phi + psi / 2),
            st * np.sin(phi + psi / 2),
        ]
    )
    return quat.T.astype(np.float32)


def hopf_to_quat_torch(
    theta: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor
) -> torch.Tensor:
    """Torch variant of :func:`hopf_to_quat`.

    Args:
        theta: Tensor of polar angles in ``[0, pi)``.
        phi: Tensor of azimuthal angles in ``[0, 2 * pi)``.
        psi: Tensor of fiber angles in ``[0, 2 * pi)``.

    Returns:
        Tensor of shape ``(..., 4)`` with scalar-first quaternions.
    """
    ct = torch.cos(theta / 2)
    st = torch.sin(theta / 2)
    return torch.stack(
        [
            ct * torch.cos(psi / 2),
            ct * torch.sin(psi / 2),
            st * torch.cos(phi + psi / 2),
            st * torch.sin(phi + psi / 2),
        ],
        dim=-1,
    ).to(torch.float32)


def grid_SO3(resol: int) -> np.ndarray:
    """Return the full SO(3) quaternion grid at a resolution level.

    Args:
        resol: Resolution level of the SO(3) grid.

    Returns:
        Array of shape ``(N, 4)`` containing the quaternion grid for the
        corresponding Hopf-fibration discretization.
    """
    theta, phi = grid_s2(resol)
    psi = grid_s1(resol)
    quat = hopf_to_quat(
        np.repeat(theta, len(psi)),
        np.repeat(phi, len(psi)),
        np.tile(psi, len(theta)),
    )
    return quat


def s2_grid_SO3(resol):
    """Return the S2-only quaternion grid with zero fiber angle.

    Args:
        resol: Resolution level of the SO(3) grid.

    Returns:
        Array of shape ``(12 * 4**resol, 4)`` containing quaternions with
        ``psi = 0`` for every S2 grid point.
    """
    theta, phi = grid_s2(resol)
    quat = hopf_to_quat(theta, phi, np.zeros((len(phi),)))
    return quat


# Neighbor lookup helpers.


def get_s1_neighbor(mini, curr_res):
    """Return candidate S1 children at the next resolution level.

    Args:
        mini: Parent S1 index at the current resolution.
        curr_res: Current resolution level.

    Returns:
        Tuple ``(psi, ind)`` where ``psi`` contains the candidate child angles
        and ``ind`` contains the matching S1 indices at ``curr_res + 1``.
    """
    Npix = 6 * 2 ** (curr_res + 1)
    dt = 2 * np.pi / Npix
    # In the Hopf construction, the nearest SO(3) neighbors at the next level
    # are not always given by the two direct S1 children alone. Keep a slightly
    # wider local stencil here and let the quaternion-distance check prune it
    # down later.
    ind = np.arange(2 * mini - 1, 2 * mini + 3)
    ind = np.mod(ind, Npix)
    return ind * dt + dt / 2, ind


def get_s2_neighbor(mini, cur_res):
    """Return the four S2 children at the next resolution level.

    Args:
        mini: Parent S2 index at the current resolution.
        cur_res: Current resolution level.

    Returns:
        Tuple ``((theta, phi), ind)`` for the four HEALPix child pixels at the
        next resolution level.
    """
    Nside = 2 ** (cur_res + 1)
    ind = np.arange(4) + 4 * mini
    return pix2ang(Nside, ind, nest=True), ind


def get_base_ind(ind, base):
    """Map flattened SO(3) grid indices to ``(s2_idx, s1_idx)`` pairs.

    Args:
        ind: Flattened SO(3) grid indices.
        base: Resolution level of the base SO(3) grid.

    Returns:
        Array or tensor of shape ``(N, 2)`` storing ``(s2_idx, s1_idx)`` pairs.
    """
    Np = 6 * 2**base
    if isinstance(ind, torch.Tensor):
        psii = ind % Np
        thetai = ind // Np
        return torch.stack((thetai, psii), dim=1)

    ind = np.asarray(ind)
    psii = ind % Np
    thetai = ind // Np
    return np.stack((thetai, psii), axis=1)


def get_neighbor(quat, s2i, s1i, cur_res):
    """Return the eight nearest SO(3) neighbors at the next resolution level.

    Args:
        quat: Reference quaternion with shape ``(4,)``.
        s2i: S2 index of the reference pose at the current resolution.
        s1i: S1 index of the reference pose at the current resolution.
        cur_res: Current resolution level.

    Returns:
        Tuple ``(quat_n, ind)`` where ``quat_n`` has shape ``(8, 4)`` and
        ``ind`` has shape ``(8, 2)`` with the corresponding ``(s2_idx, s1_idx)``
        pairs at the next resolution level.
    """
    (theta, phi), s2_nexti = get_s2_neighbor(s2i, cur_res)
    psi, s1_nexti = get_s1_neighbor(s1i, cur_res)
    # Form the 4 x 4 candidate product in vectorized form.
    quat_n = hopf_to_quat(
        np.repeat(theta, len(psi)), np.repeat(phi, len(psi)), np.tile(psi, len(theta))
    )
    ind = np.array([np.repeat(s2_nexti, len(psi)), np.tile(s1_nexti, len(theta))])
    ind = ind.T
    # Quaternions ``q`` and ``-q`` represent the same rotation.
    dists = np.minimum(
        np.sum((quat_n - quat) ** 2, axis=1), np.sum((quat_n + quat) ** 2, axis=1)
    )
    ii = np.argsort(dists)[:8]
    return quat_n[ii], ind[ii]


def subdivide_neighbors(
    quat: Union[np.ndarray, torch.Tensor],
    rot_grid_idx: Union[np.ndarray, torch.Tensor],
    cur_res: int,
    *,
    device: torch.device | None = None,
    quat_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subdivide SO(3) candidates to the next resolution level.

    Args:
        quat: Quaternions with shape ``(N, 4)``.
        rot_grid_idx: Grid indices with shape ``(N, 2)`` storing ``(s2_idx, s1_idx)``.
        cur_res: Current HEALPix resolution level.
        device: Target device for the returned tensors. Defaults to the input device.
        quat_dtype: Floating-point dtype for quaternion outputs.

    Returns:
        Tuple ``(quat, rot_grid_idx, rotmat)`` for the next resolution level.

    Notes:
        The original implementation expanded one parent at a time in Python and
        then converted the results back to tensors. This version keeps the same
        neighborhood definition but performs the 4 x 4 child expansion, distance
        evaluation, and top-k selection in batch to reduce Python overhead on
        the main refinement path.
    """
    if device is None:
        if isinstance(quat, torch.Tensor):
            device = quat.device
        elif isinstance(rot_grid_idx, torch.Tensor):
            device = rot_grid_idx.device
        else:
            device = torch.device("cpu")

    quat_t = torch.as_tensor(quat, device=device, dtype=quat_dtype)
    rot_grid_idx_t = torch.as_tensor(rot_grid_idx, device=device, dtype=torch.long)

    if rot_grid_idx_t.ndim == 3 and rot_grid_idx_t.shape[0] == 1:
        rot_grid_idx_t = rot_grid_idx_t.squeeze(0)

    N_sel = quat_t.shape[0]
    if quat_t.ndim != 2 or quat_t.shape != (N_sel, 4):
        raise ValueError(f"quat must have shape ({N_sel}, 4), got {tuple(quat_t.shape)}")
    if rot_grid_idx_t.ndim != 2 or rot_grid_idx_t.shape != (N_sel, 2):
        raise ValueError(
            f"rot_grid_idx must have shape ({N_sel}, 2), got {tuple(rot_grid_idx_t.shape)}"
        )

    s2_max = 12 * (4**cur_res)
    s1_max = 6 * (2**cur_res)
    if not torch.all((rot_grid_idx_t[:, 0] >= 0) & (rot_grid_idx_t[:, 0] < s2_max)):
        raise ValueError("s2i out of range")
    if not torch.all((rot_grid_idx_t[:, 1] >= 0) & (rot_grid_idx_t[:, 1] < s1_max)):
        raise ValueError("s1i out of range")

    if N_sel == 0:
        rot_grid_idx_t = torch.empty((0, 2), device=device, dtype=torch.long)
        rotmat_t = quaternion_to_matrix(quat_t)
        return quat_t, rot_grid_idx_t, rotmat_t

    # Each parent pose produces four S2 children and four local S1 candidates.
    # Their Cartesian product gives 16 SO(3) candidates before pruning.
    Nside_next = 2 ** (cur_res + 1)
    Npix_s1_next = 6 * 2 ** (cur_res + 1)
    dt = float(2 * np.pi / Npix_s1_next)

    s2_nexti = 4 * rot_grid_idx_t[:, 0:1] + torch.arange(4, device=device).view(1, 4)
    s1_offsets = torch.arange(-1, 3, device=device).view(1, 4)
    s1_nexti = torch.remainder(2 * rot_grid_idx_t[:, 1:2] + s1_offsets, Npix_s1_next)
    # ``s2_nexti`` and ``s1_nexti`` both have shape ``(N, 4)``.

    theta_flat, phi_flat = pix2ang_tensor(
        Nside_next,
        s2_nexti.reshape(-1),
        nest=True,
        device=device,
        dtype=quat_t.dtype,
    )
    # ``theta`` and ``phi`` have shape ``(N, 4)`` after reshaping the flattened lookup.
    theta = theta_flat.reshape(N_sel, 4)
    phi = phi_flat.reshape(N_sel, 4)
    psi = s1_nexti.to(quat_t.dtype) * dt + dt / 2  # Shape: ``(N, 4)``.

    # Broadcast the 4 x 4 child product to shape ``(N, 4, 4)`` before flattening.
    theta_grid = theta[:, :, None].expand(-1, -1, 4)
    phi_grid = phi[:, :, None].expand(-1, -1, 4)
    psi_grid = psi[:, None, :].expand(-1, 4, -1)

    # Convert all candidates to quaternions in batch, then reshape to ``(N, 16, 4)``.
    quat_candidates = hopf_to_quat_torch(
        theta_grid.reshape(-1),
        phi_grid.reshape(-1),
        psi_grid.reshape(-1),
    ).reshape(N_sel, 16, 4)

    # Store the matching ``(s2_idx, s1_idx)`` pairs with shape ``(N, 16, 2)``.
    candidate_indices = torch.stack(
        [
            s2_nexti[:, :, None].expand(-1, -1, 4),
            s1_nexti[:, None, :].expand(-1, 4, -1),
        ],
        dim=-1,
    ).reshape(N_sel, 16, 2)

    # Quaternions ``q`` and ``-q`` encode the same rotation, so compare against
    # both signs and keep the smaller distance.
    diff_pos = quat_candidates - quat_t[:, None, :]  # Shape: ``(N, 16, 4)``.
    diff_neg = quat_candidates + quat_t[:, None, :]  # Shape: ``(N, 16, 4)``.
    dists = torch.minimum(
        diff_pos.square().sum(dim=2),
        diff_neg.square().sum(dim=2),
    )  # Shape: ``(N, 16)``.

    top_k = min(8, quat_candidates.shape[1])
    # Select the closest child candidates independently for each parent pose.
    top_idx = torch.topk(dists, k=top_k, dim=1, largest=False, sorted=True).indices

    quat_t = torch.gather(
        quat_candidates,
        1,
        top_idx[:, :, None].expand(-1, -1, quat_candidates.shape[-1]),
    ).reshape(-1, 4)  # Shape: ``(N * top_k, 4)``.
    rot_grid_idx_t = torch.gather(
        candidate_indices,
        1,
        top_idx[:, :, None].expand(-1, -1, candidate_indices.shape[-1]),
    ).reshape(-1, 2)  # Shape: ``(N * top_k, 2)``.
    rotmat_t = quaternion_to_matrix(quat_t)
    return quat_t, rot_grid_idx_t, rotmat_t


try:
    with open(f"{os.path.dirname(__file__)}/healpy_grid.json") as hf:
        _GRIDS = {int(k): np.array(v).T for k, v in json.load(hf).items()}
except IOError:
    print(
        "WARNING: Couldn't load cached healpy grid; will fall back to importing healpy"
    )
    _GRIDS = None

_GRID_TENSOR_CACHE: dict[tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def pix2ang(Nside, ipix, nest=False, lonlat=False):
    if _GRIDS is not None and Nside in _GRIDS and nest and not lonlat:
        return _GRIDS[Nside][ipix].T
    else:
        try:
            import healpy
        except ImportError:
            raise RuntimeError(
                "You need to `pip install healpy` to run with non-standard grid sizes."
            )
        return healpy.pix2ang(Nside, ipix, nest=nest, lonlat=lonlat)


def pix2ang_tensor(
    Nside: int,
    ipix: Union[np.ndarray, torch.Tensor],
    *,
    nest: bool = False,
    lonlat: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch-friendly angle lookup using cached HEALPix tables when available.

    Args:
        Nside: HEALPix ``Nside`` parameter.
        ipix: HEALPix pixel indices.
        nest: Whether to interpret ``ipix`` in nested ordering.
        lonlat: Whether to return longitude/latitude instead of ``(theta, phi)``.
        device: Target device for the returned tensors.
        dtype: Floating-point dtype of the returned angle tensors.

    For standard cached resolutions, this avoids a round-trip through NumPy for
    every lookup by materializing the precomputed angle table directly on the
    requested device. For uncached resolutions, it falls back to the existing
    NumPy / healpy path and converts the result back to tensors.

    Returns:
        Tuple ``(theta, phi)`` as tensors with the same logical shape as ``ipix``.
    """
    if device is None:
        device = ipix.device if isinstance(ipix, torch.Tensor) else torch.device("cpu")

    if _GRIDS is not None and Nside in _GRIDS and nest and not lonlat:
        dev = torch.device(device)
        cache_key = (Nside, dev.type, dev.index, dtype)
        grid_t = _GRID_TENSOR_CACHE.get(cache_key)
        if grid_t is None:
            # Cache one tensor copy per (resolution, device, dtype) so repeated
            # refinement steps can reuse the same HEALPix angle table.
            grid_t = torch.from_numpy(_GRIDS[Nside]).to(device=dev, dtype=dtype)
            _GRID_TENSOR_CACHE[cache_key] = grid_t

        ipix_t = torch.as_tensor(ipix, device=dev, dtype=torch.long)
        angles = grid_t.index_select(0, ipix_t.reshape(-1)).reshape(*ipix_t.shape, 2)
        return angles[..., 0], angles[..., 1]

    if isinstance(ipix, torch.Tensor):
        ipix_np = ipix.detach().cpu().numpy()
    else:
        ipix_np = np.asarray(ipix)

    theta, phi = pix2ang(Nside, ipix_np, nest=nest, lonlat=lonlat)
    theta_t = torch.as_tensor(theta, device=device, dtype=dtype)
    phi_t = torch.as_tensor(phi, device=device, dtype=dtype)
    return theta_t, phi_t


def get_s2_k_step_neighbors(center_pix, k_steps, cur_res):
    """Return all S2 pixels within ``k_steps`` graph steps of ``center_pix``.

    Args:
        center_pix: Center S2 pixel index.
        k_steps: Neighborhood radius in graph steps.
        cur_res: Current resolution level.

    Returns:
        Tuple ``(neighbor_pixels, step_levels)`` where both arrays have shape
        ``(N,)`` and ``step_levels[i]`` stores the BFS layer for
        ``neighbor_pixels[i]``.
    """
    try:
        import healpy
    except ImportError:
        raise RuntimeError("You need to `pip install healpy` for neighbor finding.")

    Nside = 2**cur_res

    # Breadth-first search over the HEALPix neighborhood graph. This helper is
    # mostly used by the local-sampling utilities rather than the main hot path.
    visited = set()
    neighbors_by_step = {0: [center_pix]}
    queue = [(center_pix, 0)]
    visited.add(center_pix)

    while queue:
        current_pix, current_step = queue.pop(0)

        if current_step < k_steps:
            immediate_neighbors = healpy.get_all_neighbours(
                Nside, current_pix, nest=True
            )
            valid_neighbors = immediate_neighbors[immediate_neighbors != -1]

            for neighbor_pix in valid_neighbors:
                if neighbor_pix not in visited:
                    visited.add(neighbor_pix)
                    next_step = current_step + 1

                    if next_step not in neighbors_by_step:
                        neighbors_by_step[next_step] = []
                    neighbors_by_step[next_step].append(neighbor_pix)

                    queue.append((neighbor_pix, next_step))

    # Flatten the BFS layers while keeping the step count for each entry.
    all_neighbors = []
    step_levels = []
    for step in range(k_steps + 1):
        if step in neighbors_by_step:
            all_neighbors.extend(neighbors_by_step[step])
            step_levels.extend([step] * len(neighbors_by_step[step]))

    return np.array(all_neighbors), np.array(step_levels)


def sample_s2_higher_resolution(neighbor_pixels, cur_res, target_res):
    """Expand an S2 neighborhood from ``cur_res`` to ``target_res``.

    Args:
        neighbor_pixels: S2 pixel indices at the current resolution.
        cur_res: Current resolution level.
        target_res: Target resolution level.

    Returns:
        Tuple ``(high_res_pixels, (theta, phi))`` containing the expanded S2
        indices and their angles at the target resolution.
    """
    if target_res <= cur_res:
        raise ValueError("Target resolution must be higher than current resolution")

    subdivision_factor = 4 ** (target_res - cur_res)

    neighbor_pixels = np.asarray(neighbor_pixels, dtype=np.int64)
    child_offsets = np.arange(subdivision_factor, dtype=np.int64)
    high_res_pixels = (
        neighbor_pixels[:, None] * subdivision_factor + child_offsets[None, :]
    ).reshape(-1)

    Nside_target = 2**target_res
    theta, phi = pix2ang(Nside_target, high_res_pixels, nest=True)

    return high_res_pixels, (theta, phi)


def get_s2_local_sampling(center_pix, k_steps, cur_res, target_res):
    """Return a local S2 neighborhood sampled at a higher resolution.

    Args:
        center_pix: Center S2 pixel index at the current resolution.
        k_steps: Neighborhood radius in graph steps.
        cur_res: Current resolution level.
        target_res: Target resolution level used for sampling.

    Returns:
        Tuple ``(high_res_pixels, angles, neighbor_pixels)`` where
        ``high_res_pixels`` are the target-resolution samples, ``angles`` is the
        matching ``(theta, phi)`` tuple, and ``neighbor_pixels`` are the coarse
        S2 neighbors before subdivision.
    """
    # Step 1: Find the coarse k-step neighborhood on the current S2 grid.
    neighbor_pixels, step_levels = get_s2_k_step_neighbors(center_pix, k_steps, cur_res)

    # Step 2: Subdivide that neighborhood onto the target S2 resolution.
    high_res_pixels, angles = sample_s2_higher_resolution(
        neighbor_pixels, cur_res, target_res
    )

    return high_res_pixels, angles, neighbor_pixels


def get_s1_local_sampling(center_s1_idx, k_steps, cur_res, target_res):
    """Return a local S1 neighborhood sampled at a higher resolution.

    Args:
        center_s1_idx: Center S1 index at the current resolution.
        k_steps: Neighborhood radius on the periodic S1 grid.
        cur_res: Current resolution level.
        target_res: Target resolution level used for sampling.

    Returns:
        Tuple ``(high_res_s1_indices, psi_angles, neighbor_s1_indices)`` with the
        target-resolution S1 indices, their fiber angles, and the coarse S1
        neighbors before subdivision.
    """
    if target_res <= cur_res:
        raise ValueError("Target resolution must be higher than current resolution")

    # Step 1: Enumerate the local neighborhood on the coarse periodic S1 grid.
    Npix_s1_cur = 6 * 2**cur_res

    # S1 is a periodic 1D grid, so the k-step neighborhood can be written in
    # closed form without an explicit BFS.
    if k_steps == 0:
        offsets = np.array([0], dtype=np.int64)
    else:
        step_offsets = np.arange(1, k_steps + 1, dtype=np.int64)
        offsets = np.concatenate(
            ([0], np.column_stack((-step_offsets, step_offsets)).ravel())
        )

    neighbor_s1_indices = (center_s1_idx + offsets) % Npix_s1_cur
    _, unique_idx = np.unique(neighbor_s1_indices, return_index=True)
    neighbor_s1_indices = neighbor_s1_indices[np.sort(unique_idx)]

    # Step 2: Expand each coarse S1 bin into its children at the target resolution.
    subdivision_factor = 2 ** (target_res - cur_res)
    child_offsets = np.arange(subdivision_factor, dtype=np.int64)
    high_res_s1_indices = (
        neighbor_s1_indices[:, None] * subdivision_factor + child_offsets[None, :]
    ).reshape(-1)

    # Step 3: Convert target-level S1 indices back to fiber angles.
    Npix_s1_target = 6 * 2**target_res
    dt = 2 * np.pi / Npix_s1_target
    psi_angles = high_res_s1_indices * dt + dt / 2

    return high_res_s1_indices, psi_angles, neighbor_s1_indices


def get_so3_neighbor_current_res(quat, s2_idx, s1_idx, cur_res, n_neighbors=8):
    """Return SO(3) neighbors at the current resolution level.

    Args:
        quat: Reference quaternion with shape ``(4,)``.
        s2_idx: S2 index of the reference pose.
        s1_idx: S1 index of the reference pose.
        cur_res: Current resolution level.
        n_neighbors: Number of nearest neighbors to keep.

    Returns:
        Tuple ``(neighbor_quats, neighbor_indices)`` containing the selected
        quaternions and their ``(s2_idx, s1_idx)`` pairs.
    """
    try:
        import healpy
    except ImportError:
        raise RuntimeError("You need to `pip install healpy` for neighbor finding.")

    Nside = 2**cur_res
    Npix_s1 = 6 * 2**cur_res

    s2_neighbors = healpy.get_all_neighbours(Nside, s2_idx, nest=True)
    s2_neighbors = s2_neighbors[s2_neighbors != -1]
    s2_neighbors = np.append(s2_neighbors, s2_idx)

    # Use a small local S1 stencil around the current fiber index. As in
    # ``get_s1_neighbor()``, the local Hopf geometry is not captured by the two
    # direct children alone.
    s1_offsets = np.arange(-2, 3)
    s1_neighbors = (s1_idx + s1_offsets) % Npix_s1

    s2_mesh, s1_mesh = np.meshgrid(s2_neighbors, s1_neighbors, indexing="ij")
    candidate_indices = np.column_stack([s2_mesh.ravel(), s1_mesh.ravel()])
    # ``candidate_indices`` has shape ``(N_candidates, 2)``.

    s2_indices = candidate_indices[:, 0]
    s1_indices = candidate_indices[:, 1]

    theta, phi = pix2ang(Nside, s2_indices, nest=True)
    dt = 2 * np.pi / Npix_s1
    psi = s1_indices * dt + dt / 2

    # Convert all candidates to quaternions in vectorized form.
    candidate_quats = hopf_to_quat(theta, phi, psi)

    center_mask = ~(
        (candidate_indices[:, 0] == s2_idx) & (candidate_indices[:, 1] == s1_idx)
    )
    candidate_quats = candidate_quats[center_mask]
    candidate_indices = candidate_indices[center_mask]

    if len(candidate_quats) == 0:
        return np.array([]), np.array([])

    if quat.ndim > 1:
        quat = quat.flatten()[:4]

    diff_pos = candidate_quats - quat[None, :]  # Shape: ``(N_candidates, 4)``.
    diff_neg = candidate_quats + quat[None, :]  # Shape: ``(N_candidates, 4)``.

    dist_pos = np.sum(diff_pos**2, axis=1)
    dist_neg = np.sum(diff_neg**2, axis=1)

    dists = np.minimum(dist_pos, dist_neg)

    n_to_select = min(n_neighbors, len(candidate_quats))
    if n_to_select == len(candidate_quats):
        closest_sorted = np.argsort(dists)
    else:
        closest_indices = np.argpartition(dists, n_to_select - 1)[:n_to_select]
        closest_sorted = closest_indices[np.argsort(dists[closest_indices])]

    neighbor_quats = candidate_quats[closest_sorted]
    neighbor_indices = candidate_indices[closest_sorted]

    return neighbor_quats, neighbor_indices


def get_so3_k_step_neighbors(
    center_quat, center_s2_idx, center_s1_idx, k_steps, cur_res
):
    """Return the SO(3) neighborhood within ``k_steps`` BFS layers.

    Args:
        center_quat: Center quaternion with shape ``(4,)``.
        center_s2_idx: Center S2 index.
        center_s1_idx: Center S1 index.
        k_steps: Neighborhood radius in BFS layers.
        cur_res: Current resolution level.

    Returns:
        Tuple ``(neighbor_quats, neighbor_indices, step_levels)`` storing the
        discovered quaternions, their grid indices, and the BFS layer of each
        entry.
    """
    visited = set()
    neighbors_by_step = {0: [(center_quat, center_s2_idx, center_s1_idx)]}
    queue = [(center_quat, center_s2_idx, center_s1_idx, 0)]
    visited.add((center_s2_idx, center_s1_idx))

    while queue:
        current_quat, current_s2, current_s1, current_step = queue.pop(0)

        if current_step < k_steps:
            neighbor_quats, neighbor_indices = get_so3_neighbor_current_res(
                current_quat, current_s2, current_s1, cur_res
            )

            for n_quat, n_ind in zip(neighbor_quats, neighbor_indices):
                n_s2, n_s1 = n_ind[0], n_ind[1]

                if (n_s2, n_s1) not in visited:
                    visited.add((n_s2, n_s1))
                    next_step = current_step + 1

                    if next_step not in neighbors_by_step:
                        neighbors_by_step[next_step] = []
                    neighbors_by_step[next_step].append((n_quat, n_s2, n_s1))

                    queue.append((n_quat, n_s2, n_s1, next_step))

    all_quats = []
    all_indices = []
    step_levels = []

    for step in range(k_steps + 1):
        if step in neighbors_by_step:
            for quat, s2_idx, s1_idx in neighbors_by_step[step]:
                all_quats.append(quat)
                all_indices.append([s2_idx, s1_idx])
                step_levels.append(step)

    return np.array(all_quats), np.array(all_indices), np.array(step_levels)


def sample_so3_higher_resolution(neighbor_indices, cur_res, target_res):
    """Expand an SO(3) neighborhood from ``cur_res`` to ``target_res``.

    Args:
        neighbor_indices: Array of shape ``(N, 2)`` storing coarse
            ``(s2_idx, s1_idx)`` pairs.
        cur_res: Current resolution level.
        target_res: Target resolution level.

    Returns:
        Tuple ``(high_res_indices, high_res_quats, high_res_rots, angles)``
        containing the expanded indices, quaternions, rotation matrices, and the
        corresponding ``(theta, phi, psi)`` angles.
    """
    if target_res <= cur_res:
        raise ValueError("Target resolution must be higher than current resolution")

    s2_subdivision_factor = 4 ** (target_res - cur_res)
    s1_subdivision_factor = 2 ** (target_res - cur_res)

    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    s2_offsets, s1_offsets = np.meshgrid(
        np.arange(s2_subdivision_factor, dtype=np.int64),
        np.arange(s1_subdivision_factor, dtype=np.int64),
        indexing="ij",
    )
    high_res_indices = np.stack(
        [
            neighbor_indices[:, 0, None, None] * s2_subdivision_factor
            + s2_offsets[None, :, :],
            neighbor_indices[:, 1, None, None] * s1_subdivision_factor
            + s1_offsets[None, :, :],
        ],
        axis=-1,
    ).reshape(-1, 2)

    s2_indices = high_res_indices[:, 0]
    s1_indices = high_res_indices[:, 1]

    Nside_target = 2**target_res
    theta, phi = pix2ang(Nside_target, s2_indices, nest=True)

    Npix_s1_target = 6 * 2**target_res
    dt = 2 * np.pi / Npix_s1_target
    psi = s1_indices * dt + dt / 2

    high_res_quats = hopf_to_quat(theta, phi, psi)

    high_res_rots = quaternion_to_matrix(torch.from_numpy(high_res_quats))

    return high_res_indices, high_res_quats, high_res_rots, (theta, phi, psi)


def get_so3_local_sampling(
    center_quat, center_s2_idx, center_s1_idx, k_steps, cur_res, target_res
):
    """Return local SO(3) samples around a center pose.

    Args:
        center_quat: Center quaternion with shape ``(4,)``.
        center_s2_idx: Center S2 index at the current resolution.
        center_s1_idx: Center S1 index at the current resolution.
        k_steps: Neighborhood radius in SO(3) graph steps.
        cur_res: Current resolution level.
        target_res: Target resolution level used for sampling.

    Returns:
        Tuple ``(indices, quats, rots, angles)`` describing the local SO(3)
        samples at the target resolution.
    """
    # Step 1: Find the SO(3) neighborhood at the current resolution.
    neighbor_quats, neighbor_indices, step_levels = get_so3_k_step_neighbors(
        center_quat, center_s2_idx, center_s1_idx, k_steps, cur_res
    )
    if target_res > cur_res:
        # Step 2: Subdivide the coarse neighborhood onto the target resolution.
        indices, quats, rots, angles = sample_so3_higher_resolution(
            neighbor_indices, cur_res, target_res
        )
    elif target_res == cur_res:
        # Step 2: Reuse the coarse neighborhood directly when no subdivision is needed.
        indices = neighbor_indices
        quats = neighbor_quats
        rots = quaternion_to_matrix(torch.from_numpy(neighbor_quats))
        s2_indices = indices[:, 0]
        s1_indices = indices[:, 1]

        # Step 3: Recover the angle parameterization for the selected grid points.
        Nside_target = 2**target_res
        theta, phi = pix2ang(Nside_target, s2_indices, nest=True)

        Npix_s1_target = 6 * 2**target_res
        dt = 2 * np.pi / Npix_s1_target
        psi = s1_indices * dt + dt / 2

        angles = (theta, phi, psi)
    else:
        raise ValueError("Current resolution cannot be higher than target resolution")

    return indices, quats, rots, angles


def get_local_s2_poses(
    center_s2_pix, center_s1_idx, k_steps, cur_res, target_res
):
    """Generate local pose samples around a center pose.

    Args:
        center_s2_pix: Center S2 pixel index.
        center_s1_idx: Center S1 index.
        k_steps: Neighborhood radius.
        cur_res: Current resolution level.
        target_res: Target sampling resolution.

    Returns:
        Tuple ``(pose_samples, quaternions)`` where ``pose_samples`` stores
        ``(s2_idx, s1_idx)`` pairs and ``quaternions`` stores the corresponding
        rotations.
    """
    # Step 1: Sample the local S2 neighborhood at the target resolution.
    s2_high_res_pixels, (theta, phi), _ = get_s2_local_sampling(
        center_s2_pix, k_steps, cur_res, target_res
    )

    # Step 2: Sample the local S1 neighborhood at the target resolution.
    s1_high_res_indices, psi_angles, _ = get_s1_local_sampling(
        center_s1_idx, k_steps, cur_res, target_res
    )

    # Step 3: Form the Cartesian product of the S2 and S1 neighborhoods.
    num_s1 = len(s1_high_res_indices)
    num_s2 = len(s2_high_res_pixels)
    pose_samples = np.column_stack(
        [
            np.repeat(s2_high_res_pixels, num_s1),
            np.tile(s1_high_res_indices, num_s2),
        ]
    )
    all_theta = np.repeat(theta, num_s1)
    all_phi = np.repeat(phi, num_s1)
    all_psi = np.tile(psi_angles, num_s2)

    # Step 4: Convert the local pose samples to quaternions.
    quaternions = hopf_to_quat(all_theta, all_phi, all_psi)

    return pose_samples, quaternions


def get_quat_from_ind(resol, s2i, s1i):
    """Convert SO(3) grid indices to quaternions at a given resolution.

    Args:
        resol: Resolution level of the SO(3) grid.
        s2i: S2 indices.
        s1i: S1 indices with the same shape as ``s2i``.

    Returns:
        Array of shape ``(..., 4)`` containing the corresponding quaternions.
    """
    s2i = np.asarray(s2i)
    s1i = np.asarray(s1i)

    if s2i.shape != s1i.shape:
        raise ValueError("s2i and s1i must have the same shape")

    original_shape = s2i.shape
    s2i_flat = s2i.flatten()
    s1i_flat = s1i.flatten()

    Nside = 2**resol
    theta, phi = pix2ang(Nside, s2i_flat, nest=True)

    Npix_s1 = 6 * 2**resol
    dt = 2 * np.pi / Npix_s1
    psi = s1i_flat * dt + dt / 2

    quaternions = hopf_to_quat(theta, phi, psi)

    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)

    if original_shape:
        quaternions = quaternions.reshape(*original_shape, 4)

    return quaternions


def euler_local_sampling(eulers, target_res, k_steps=1):
    """Sample a local Euler-angle neighborhood using grid-derived step sizes.

    Args:
        eulers: Tensor or array with shape ``(3,)`` or ``(N, 3)`` storing
            ``[phi, theta, psi]`` angles.
        target_res: Resolution level used to derive angular step sizes.
        k_steps: Number of steps in each direction.

    Returns:
        Tuple ``(candidate_eulers, candidate_quats, candidate_rots)``.
    """
    if isinstance(eulers, np.ndarray):
        eulers = torch.from_numpy(eulers.astype(np.float32))

    if eulers.ndim == 1:
        eulers = eulers.unsqueeze(0)

    euler_angles = eulers

    Nside_target = 2**target_res
    theta_step = np.sqrt(4 * np.pi / (12 * Nside_target**2))
    phi_step = theta_step

    Npix_s1_target = 6 * 2**target_res
    psi_step = 2 * np.pi / Npix_s1_target

    k_range = torch.arange(-k_steps, k_steps + 1, dtype=torch.float32)
    k_phi_grid, k_theta_grid, k_psi_grid = torch.meshgrid(
        k_range, k_range, k_range, indexing="ij"
    )
    # Flatten the 3D offset grid to ``(Q,)`` so it can broadcast against ``(N, 1)`` centers.
    k_phi_flat = k_phi_grid.reshape(-1)
    k_theta_flat = k_theta_grid.reshape(-1)
    k_psi_flat = k_psi_grid.reshape(-1)

    device = eulers.device
    k_phi_flat = k_phi_flat.to(device)
    k_theta_flat = k_theta_flat.to(device)
    k_psi_flat = k_psi_flat.to(device)

    n_inputs = euler_angles.shape[0]
    n_offsets = k_phi_flat.shape[0]

    phi_center = euler_angles[:, 0].unsqueeze(1)  # Shape: ``(N, 1)``.
    theta_center = euler_angles[:, 1].unsqueeze(1)  # Shape: ``(N, 1)``.
    psi_center = euler_angles[:, 2].unsqueeze(1)  # Shape: ``(N, 1)``.

    # Broadcast center angles against all offset combinations to get ``(N, Q)`` tensors.
    phi_candidates = phi_center + k_phi_flat.unsqueeze(0) * phi_step
    theta_candidates = theta_center + k_theta_flat.unsqueeze(0) * theta_step
    psi_candidates = psi_center + k_psi_flat.unsqueeze(0) * psi_step

    phi_candidates = phi_candidates % (2 * np.pi)
    phi_candidates = torch.where(
        phi_candidates > np.pi, phi_candidates - 2 * np.pi, phi_candidates
    )

    psi_candidates = psi_candidates % (2 * np.pi)
    psi_candidates = torch.where(
        psi_candidates > np.pi, psi_candidates - 2 * np.pi, psi_candidates
    )

    theta_candidates = torch.clamp(theta_candidates, 0, np.pi)

    candidate_eulers = torch.stack(
        [phi_candidates, theta_candidates, psi_candidates], dim=2
    )
    candidate_eulers = candidate_eulers.reshape(-1, 3)  # Shape: ``(N * Q, 3)``.

    candidate_rots = euler_to_matrix(candidate_eulers, device=device)  # Shape: ``(N * Q, 3, 3)``.

    candidate_quats = matrix_to_quaternion(candidate_rots)  # Shape: ``(N * Q, 4)``.

    return candidate_eulers, candidate_quats, candidate_rots