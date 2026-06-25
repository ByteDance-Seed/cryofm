"""
Hierarchical translation-grid utilities used by pose search.

This implementation is adapted from cryoDRGN-style shift-grid logic, but has
been substantially reworked from earlier NumPy-style code and is now
implemented directly with PyTorch tensors.
"""

import torch


def _infer_device(*values) -> torch.device | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def _as_tensor(value, *, dtype=None, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if dtype is not None or device is not None:
            return value.to(
                dtype=dtype if dtype is not None else value.dtype,
                device=device if device is not None else value.device,
            )
        return value
    return torch.as_tensor(value, dtype=dtype, device=device)


def _as_scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def grid_1d(resol: int, extent: int, ngrid: int, shift: int = 0) -> torch.Tensor:
    Npix = ngrid * 2**resol
    dt = 2 * extent / Npix
    grid = torch.arange(Npix, dtype=torch.float32) * dt + dt / 2 - extent + shift
    return grid


def get_1d_shift(
    mini, resol: int, extent: int, ngrid: int, shift: int = 0
) -> torch.Tensor:
    Npix = ngrid * 2**resol
    dt = 2 * extent / Npix
    mini = _as_tensor(mini, dtype=torch.float32, device=_infer_device(mini))
    grid = mini * dt + dt / 2 - extent + shift
    return grid


def grid_2d(
    resol: int, extent: int, ngrid: int, xshift: int = 0, yshift: int = 0
) -> torch.Tensor:
    x = grid_1d(resol, extent, ngrid, shift=xshift)
    y = grid_1d(resol, extent, ngrid, shift=yshift)
    # convention: x is fast dim, y is slow dim
    x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack((x_grid, y_grid), dim=-1)
    return grid.reshape(-1, 2)


def base_shift_grid(
    resol: int, extent: int, ngrid: int, xshift: int = 0, yshift: int = 0
) -> torch.Tensor:
    return grid_2d(resol, extent, ngrid, xshift, yshift)


# Neighbor Finding
def get_1d_neighbor(mini, cur_res, extent, ngrid):
    Npix = ngrid * 2 ** (cur_res + 1)
    dt = 2 * extent / Npix
    device = _infer_device(mini)
    mini = _as_tensor(mini, dtype=torch.long, device=device)
    ind = torch.stack((2 * mini, 2 * mini + 1), dim=-1)
    return dt * ind + dt / 2 - extent, ind


def get_base_ind(ind, ngrid):
    """
    Only get 2D indices on the base grid.
    """
    device = _infer_device(ind)
    ind = _as_tensor(ind, dtype=torch.long, device=device)
    xi = ind % ngrid
    yi = ind // ngrid
    return torch.stack((xi, yi), dim=-1)


def get_ind(ind, Npix):
    """
    Get 2D indices on the grid with ``Npix`` pixels per side.
    """
    device = _infer_device(ind)
    ind = _as_tensor(ind, dtype=torch.long, device=device)
    xi = ind % Npix
    yi = ind // Npix
    return torch.stack((xi, yi), dim=-1)


def get_neighbor(xi, yi, cur_res, extent, ngrid):
    """
    Return the 4 nearest neighbors at the next resolution level
    """
    x_next, xii = get_1d_neighbor(xi, cur_res, extent, ngrid)
    y_next, yii = get_1d_neighbor(yi, cur_res, extent, ngrid)
    t_next = torch.stack(
        (
            torch.stack((x_next[..., 0], y_next[..., 0]), dim=-1),
            torch.stack((x_next[..., 1], y_next[..., 0]), dim=-1),
            torch.stack((x_next[..., 0], y_next[..., 1]), dim=-1),
            torch.stack((x_next[..., 1], y_next[..., 1]), dim=-1),
        ),
        dim=-2,
    )
    ind_next = torch.stack(
        (
            torch.stack((xii[..., 0], yii[..., 0]), dim=-1),
            torch.stack((xii[..., 1], yii[..., 0]), dim=-1),
            torch.stack((xii[..., 0], yii[..., 1]), dim=-1),
            torch.stack((xii[..., 1], yii[..., 1]), dim=-1),
        ),
        dim=-2,
    )
    return t_next, ind_next


def get_2d_neighbor_current_res(xi, yi, cur_res, extent, ngrid):
    """
    Get 2D Euclidean neighbors at the CURRENT resolution level

    Args:
        xi: int, x index at current resolution
        yi: int, y index at current resolution
        cur_res: int, current resolution level
        extent: int, spatial extent
        ngrid: int, base grid size

    Returns:
        neighbor_coords: (N, 2) array of (x, y) coordinates
        neighbor_indices: (N, 2) array of (xi, yi) indices
    """
    xi = _as_scalar_int(xi)
    yi = _as_scalar_int(yi)
    Npix_1d = ngrid * 2**cur_res

    # Get immediate neighbors in 2D grid (8-connected neighborhood)
    xi_neighbors = []
    yi_neighbors = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:  # Skip center point
                continue
            new_xi = (xi + dx) % Npix_1d  # Periodic boundary
            new_yi = (yi + dy) % Npix_1d  # Periodic boundary
            xi_neighbors.append(new_xi)
            yi_neighbors.append(new_yi)

    xi_neighbors = torch.tensor(xi_neighbors, dtype=torch.long)
    yi_neighbors = torch.tensor(yi_neighbors, dtype=torch.long)

    # Convert indices to coordinates
    dt = 2 * extent / Npix_1d
    x_coords = xi_neighbors * dt + dt / 2 - extent
    y_coords = yi_neighbors * dt + dt / 2 - extent

    neighbor_coords = torch.stack((x_coords, y_coords), dim=-1)
    neighbor_indices = torch.stack((xi_neighbors, yi_neighbors), dim=-1)

    return neighbor_coords, neighbor_indices


def get_2d_k_step_neighbors(
    center_xi, center_yi, k_steps, cur_res, extent, ngrid
):
    """
    Find all 2D grid points within k steps of the center using BFS

    Args:
        center_xi: int, center x index
        center_yi: int, center y index
        k_steps: int, maximum number of steps
        cur_res: int, current resolution level
        extent: int, spatial extent
        ngrid: int, base grid size

    Returns:
        neighbor_coords: (N, 2) array of (x, y) coordinates
        neighbor_indices: (N, 2) array of (xi, yi) indices
        step_levels: (N,) array of step distances
    """
    center_xi = _as_scalar_int(center_xi)
    center_yi = _as_scalar_int(center_yi)

    visited = set()
    neighbors_by_step = {0: [(center_xi, center_yi)]}
    queue = [(center_xi, center_yi, 0)]
    visited.add((center_xi, center_yi))

    Npix_1d = ngrid * 2**cur_res

    while queue:
        current_xi, current_yi, current_step = queue.pop(0)

        if current_step < k_steps:
            # Get immediate neighbors (4-connected)
            for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:

                neighbor_xi = (current_xi + dx) % Npix_1d
                neighbor_yi = (current_yi + dy) % Npix_1d

                if (neighbor_xi, neighbor_yi) not in visited:
                    visited.add((neighbor_xi, neighbor_yi))
                    next_step = current_step + 1

                    if next_step not in neighbors_by_step:
                        neighbors_by_step[next_step] = []
                    neighbors_by_step[next_step].append((neighbor_xi, neighbor_yi))

                    queue.append((neighbor_xi, neighbor_yi, next_step))

    # Flatten all neighbors
    all_indices = []
    step_levels = []

    for step in range(k_steps + 1):
        if step in neighbors_by_step:
            for xi, yi in neighbors_by_step[step]:
                all_indices.append([xi, yi])
                step_levels.append(step)

    all_indices = torch.tensor(all_indices, dtype=torch.long)
    step_levels = torch.tensor(step_levels, dtype=torch.long)

    # Convert to coordinates
    dt = 2 * extent / Npix_1d
    x_coords = all_indices[:, 0] * dt + dt / 2 - extent
    y_coords = all_indices[:, 1] * dt + dt / 2 - extent
    neighbor_coords = torch.stack((x_coords, y_coords), dim=-1)

    return neighbor_coords, all_indices, step_levels


def sample_2d_higher_resolution(neighbor_indices, cur_res, target_res, extent, ngrid):
    """
    Sample 2D grid at higher resolution within the k-step neighborhood

    Args:
        neighbor_indices: (N, 2) array of (xi, yi) at current resolution
        cur_res: int, current resolution level
        target_res: int, target higher resolution level
        extent: int, spatial extent
        ngrid: int, base grid size

    Returns:
        high_res_coords: (M, 2) array of (x, y) coordinates at target resolution
        high_res_indices: (M, 2) array of (xi, yi) indices at target resolution
    """
    if target_res <= cur_res:
        raise ValueError("Target resolution must be higher than current resolution")

    # Calculate subdivision factor
    subdivision_factor = 2 ** (target_res - cur_res)
    device = _infer_device(neighbor_indices)
    neighbor_indices = _as_tensor(neighbor_indices, dtype=torch.long, device=device)

    offsets = torch.arange(subdivision_factor, device=neighbor_indices.device)
    x_offset, y_offset = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_pairs = torch.stack((x_offset, y_offset), dim=-1).reshape(-1, 2)
    high_res_indices = (
        neighbor_indices[:, None, :] * subdivision_factor + offset_pairs[None, :, :]
    ).reshape(-1, 2)

    # Convert to coordinates
    Npix_1d_target = ngrid * 2**target_res
    dt = 2 * extent / Npix_1d_target
    x_coords = high_res_indices[:, 0] * dt + dt / 2 - extent
    y_coords = high_res_indices[:, 1] * dt + dt / 2 - extent
    high_res_coords = torch.stack((x_coords, y_coords), dim=-1)

    return high_res_coords, high_res_indices


def get_2d_local_sampling(
    center_xi, center_yi, k_steps, cur_res, target_res, extent, ngrid
):
    """
    Complete 2D local sampling pipeline: find k-step neighbors and sample at higher resolution

    Args:
        center_xi: int, center x index at current resolution
        center_yi: int, center y index at current resolution
        k_steps: int, neighborhood radius in steps
        cur_res: int, current resolution level
        target_res: int, target sampling resolution
        extent: int, spatial extent
        ngrid: int, base grid size

    Returns:
        high_res_coords: (N, 2) array of (x, y) coordinates at target resolution
        high_res_indices: (N, 2) array of (xi, yi) indices at target resolution
        original_neighbors: original k-step neighbors at current resolution
    """
    # Step 1: Find k-step neighbors at current resolutio
    neighbor_coords, neighbor_indices, _ = get_2d_k_step_neighbors(
        center_xi, center_yi, k_steps, cur_res, extent, ngrid
    )
    if cur_res < target_res:
        # Step 2: Sample at higher resolution within this neighborhood
        coords, indices = sample_2d_higher_resolution(
            neighbor_indices, cur_res, target_res, extent, ngrid
        )
    elif cur_res == target_res:
        coords = neighbor_coords
        indices = neighbor_indices
    else:
        raise ValueError("Current resolution cannot be higher than target resolution")

    return coords, indices


def get_trans_from_ind(resol, xi, yi, extent, ngrid):
    """
    Convert 2D translation grid indices to translation coordinates at given resolution

    Args:
        resol: int, resolution level
        xi: int or array, x index in 2D grid
        yi: int or array, y index in 2D grid
        extent: int, spatial extent
        ngrid: int, base grid size

    Returns:
        translations: (N, 2) array of (x, y) translation coordinates
    """
    # Handle single indices or arrays
    device = _infer_device(xi, yi)
    xi = _as_tensor(xi, dtype=torch.long, device=device)
    yi = _as_tensor(yi, dtype=torch.long, device=device)

    # Ensure they're the same shape
    if xi.shape != yi.shape:
        raise ValueError("xi and yi must have the same shape")

    original_shape = xi.shape
    xi_flat = xi.flatten()
    yi_flat = yi.flatten()

    # Calculate grid parameters at given resolution
    Npix_1d = ngrid * 2**resol
    dt = 2 * extent / Npix_1d

    # Convert indices to coordinates
    x_coords = xi_flat * dt + dt / 2 - extent
    y_coords = yi_flat * dt + dt / 2 - extent

    # Stack into (N, 2) array
    translations = torch.stack((x_coords, y_coords), dim=-1)

    # Reshape back to original shape if needed
    if original_shape:
        translations = translations.reshape(*original_shape, 2)

    return translations


def subdivide_2Dgrid(trans, ind2d, cur_res: int, t_extent: int, t_ngrid: int):
    """
    Find the neighbors of translations for the next resolution

    Parameters
    ----------
    trans : (N,2) translations
    ind : (N,2) trans 2d index
    cur_res : int, current healpix order

    NOTE: The actual current resolution for translation is the current healpix order - 1

    Returns
    -------
    trans : (N, 4, 2) neighbors of translations
    trans_idx : (N, 4, 2) 2d index of trans on the next resolution

    """
    trans_dtype = trans.dtype if trans.is_floating_point() else torch.float32
    ind2d = _as_tensor(ind2d, dtype=torch.long, device=trans.device)
    next_trans, trans_idx = get_neighbor(
        ind2d[:, 0], ind2d[:, 1], cur_res - 1, t_extent, t_ngrid
    )
    next_trans = next_trans.to(device=trans.device, dtype=trans_dtype)
    trans_idx = trans_idx.to(device=trans.device, dtype=torch.long)
    return next_trans, trans_idx


def translation_local_sampling(
    translations, target_res, extent, ngrid, k_steps=1
):
    """
    Local sampling around given translation coordinates using grid steps from target_res

    Args:
        translations: torch.Tensor (2,) or (N, 2) translation coordinates [x, y] to search around
        target_res: int, target resolution for grid step size
        extent: int, spatial extent
        ngrid: int, base grid size
        k_steps: int, number of steps to search in each direction (default 1)

    Returns:
        candidate_translations: torch.Tensor (M, 2) of candidate translation coordinates [x, y]

    Note:
        Grid step is determined by target_res: dt = 2 * extent / (ngrid * 2^target_res)
    """
    # Convert to torch if needed
    if not isinstance(translations, torch.Tensor):
        translations = torch.as_tensor(translations, dtype=torch.float32)
    else:
        translations = translations.to(dtype=torch.float32)

    if translations.ndim == 1:
        translations = translations.unsqueeze(0)

    # Use input translations directly
    trans_coords = translations  # (N, 2) [x, y]

    # Calculate grid step for target resolution
    Npix_1d_target = ngrid * 2**target_res
    dt = 2 * extent / Npix_1d_target

    # Generate perturbation offsets (on CPU, then move to device)
    k_range = torch.arange(-k_steps, k_steps + 1, dtype=torch.float32)
    k_x_grid, k_y_grid = torch.meshgrid(k_range, k_range, indexing="ij")
    k_x_flat = k_x_grid.reshape(-1)  # Shape: (n_offsets,)
    k_y_flat = k_y_grid.reshape(-1)

    # Move to same device as translations
    device = translations.device
    k_x_flat = k_x_flat.to(device)
    k_y_flat = k_y_flat.to(device)

    # Generate candidates for all input translations at once
    # trans_coords: (N, 2), k_flat: (n_offsets,)
    # We want: (N, n_offsets, 2)
    # Expand dimensions for broadcasting
    x_center = trans_coords[:, 0].unsqueeze(1)  # (N, 1)
    y_center = trans_coords[:, 1].unsqueeze(1)  # (N, 1)

    # Perturb coordinates (vectorized over all inputs)
    x_candidates = x_center + k_x_flat.unsqueeze(0) * dt  # (N, n_offsets)
    y_candidates = y_center + k_y_flat.unsqueeze(0) * dt  # (N, n_offsets)

    # Clamp to extent boundaries [-extent, extent] (vectorized)
    # x_candidates = torch.clamp(x_candidates, -extent, extent)
    # y_candidates = torch.clamp(y_candidates, -extent, extent)

    # Stack and flatten: (N, n_offsets, 2) -> (N*n_offsets, 2)
    candidate_translations = torch.stack([x_candidates, y_candidates], dim=2)
    candidate_translations = candidate_translations.reshape(-1, 2)

    return candidate_translations