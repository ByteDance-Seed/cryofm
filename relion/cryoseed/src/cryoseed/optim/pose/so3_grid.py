"""
Implementation of Yershova et al. "Generating uniform incremental
grids on SO(3) using the Hopf fribration"
"""

import json
import os

from typing import Tuple, Union

import numpy as np
import torch

from cryoseed.cryoem.rotation import quaternion_to_matrix, matrix_to_quaternion, matrix_to_euler, euler_to_matrix


def grid_s1(resol: int) -> np.ndarray:
    Npix = 6 * 2**resol
    dt = 2 * np.pi / Npix
    grid = np.arange(Npix) * dt + dt / 2
    return grid


def grid_s2(resol: int):
    Nside = 2**resol
    Npix = 12 * Nside * Nside
    theta, phi = pix2ang(Nside, np.arange(Npix), nest=True)
    return theta, phi


def hopf_to_quat(theta, phi, psi) -> np.ndarray:
    """
    Hopf coordinates to quaternions
    theta: [0,pi)
    phi: [0, 2pi)
    psi: [0, 2pi)
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


def grid_SO3(resol: int) -> np.ndarray:
    theta, phi = grid_s2(resol)
    psi = grid_s1(resol)
    quat = hopf_to_quat(
        np.repeat(theta, len(psi)),  # repeats each element by len(psi)
        np.repeat(phi, len(psi)),  # repeats each element by len(psi)
        np.tile(psi, len(theta)),
    )  # tiles the array len(theta) times
    return quat  # hmm convert to rot matrix?


def s2_grid_SO3(resol):
    theta, phi = grid_s2(resol)
    quat = hopf_to_quat(theta, phi, np.zeros((len(phi),)))
    return quat


# Neighbor finding


def get_s1_neighbor(mini, curr_res):
    """
    Return the 2 nearest neighbors on S1 at the next resolution level
    """
    Npix = 6 * 2 ** (curr_res + 1)
    dt = 2 * np.pi / Npix
    # return np.array([2*mini, 2*mini+1])*dt + dt/2
    # the fiber bundle grid on SO3 is weird
    # the next resolution level's nearest neighbors in SO3 are not
    # necessarily the nearest neighbor grid points in S1
    # include the 13 neighbors for now... eventually learn/memoize the mapping
    ind = np.arange(2 * mini - 1, 2 * mini + 3)
    ind = np.mod(ind, Npix)
    # if ind[0] < 0:
    #     ind[0] += Npix
    return ind * dt + dt / 2, ind


def get_s2_neighbor(mini, cur_res):
    """
    Return the 4 nearest neighbors on S2 at the next resolution level
    """
    Nside = 2 ** (cur_res + 1)
    ind = np.arange(4) + 4 * mini
    return pix2ang(Nside, ind, nest=True), ind


def get_base_ind(ind, base):
    """
    Return the corresponding S2 and S1 grid index for an index on the base SO3 grid
    """
    Np = 6 * 2**base
    psii = ind % Np
    thetai = ind // Np
    return np.stack((thetai, psii), axis=1)


def get_neighbor(quat, s2i, s1i, cur_res):
    """
    Return the 8 nearest neighbors on SO3 at the next resolution level
    """
    (theta, phi), s2_nexti = get_s2_neighbor(s2i, cur_res)
    psi, s1_nexti = get_s1_neighbor(s1i, cur_res)
    quat_n = hopf_to_quat(
        np.repeat(theta, len(psi)), np.repeat(phi, len(psi)), np.tile(psi, len(theta))
    )
    ind = np.array([np.repeat(s2_nexti, len(psi)), np.tile(s1_nexti, len(theta))])
    ind = ind.T
    # find the 8 nearest neighbors of 16 possible points
    # need to check distance from both +q and -q
    dists = np.minimum(
        np.sum((quat_n - quat) ** 2, axis=1), np.sum((quat_n + quat) ** 2, axis=1)
    )
    ii = np.argsort(dists)[:8]
    return quat_n[ii], ind[ii]


try:
    with open(f"{os.path.dirname(__file__)}/healpy_grid.json") as hf:
        _GRIDS = {int(k): np.array(v).T for k, v in json.load(hf).items()}
except IOError:
    print(
        "WARNING: Couldn't load cached healpy grid; will fall back to importing healpy"
    )
    _GRIDS = None


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


def get_s2_k_step_neighbors(center_pix, k_steps, cur_res):
    """
    Find all S2 pixels within k steps of the center pixel using BFS

    Args:
        center_pix: int, center pixel index
        k_steps: int, maximum number of steps
        cur_res: int, current resolution level

    Returns:
        neighbor_pixels: list of pixel indices within k steps
        step_levels: list of step distances for each neighbor
    """
    try:
        import healpy
    except ImportError:
        raise RuntimeError("You need to `pip install healpy` for neighbor finding.")

    Nside = 2**cur_res

    # BFS to find k-step neighbors
    visited = set()
    neighbors_by_step = {0: [center_pix]}  # step -> list of pixels at that step
    queue = [(center_pix, 0)]  # (pixel_index, current_step)
    visited.add(center_pix)

    while queue:
        current_pix, current_step = queue.pop(0)

        if current_step < k_steps:
            # Get immediate neighbors of current pixel
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

    # Flatten all neighbors and their step levels
    all_neighbors = []
    step_levels = []
    # Negiable ???
    for step in range(k_steps + 1):
        if step in neighbors_by_step:
            all_neighbors.extend(neighbors_by_step[step])
            step_levels.extend([step] * len(neighbors_by_step[step]))

    return np.array(all_neighbors), np.array(step_levels)


def sample_s2_higher_resolution(neighbor_pixels, cur_res, target_res):
    """
    Sample S2 pixels at higher resolution within the k-step neighborhood

    Args:
        neighbor_pixels: array of pixel indices at current resolution
        cur_res: int, current resolution level
        target_res: int, target higher resolution level

    Returns:
        high_res_pixels: array of pixel indices at target resolution
        angles: (theta, phi) coordinates of high-res pixels
    """
    if target_res <= cur_res:
        raise ValueError("Target resolution must be higher than current resolution")

    # Calculate subdivision factor
    subdivision_factor = 4 ** (target_res - cur_res)

    high_res_pixels = []

    for pix in neighbor_pixels:
        # Each pixel at cur_res maps to subdivision_factor pixels at target_res
        start_pix = pix * subdivision_factor
        end_pix = start_pix + subdivision_factor
        high_res_pixels.extend(range(start_pix, end_pix))

    high_res_pixels = np.array(high_res_pixels)

    # Get angles for high-resolution pixels
    Nside_target = 2**target_res
    theta, phi = pix2ang(Nside_target, high_res_pixels, nest=True)

    return high_res_pixels, (theta, phi)


def get_s2_local_sampling(center_pix, k_steps, cur_res, target_res):
    """
    Complete pipeline: find k-step neighbors and sample at higher resolution

    Args:
        center_pix: int, center pixel index at current resolution
        k_steps: int, neighborhood radius in steps
        cur_res: int, current resolution level
        target_res: int, target sampling resolution

    Returns:
        high_res_pixels: pixel indices at target resolution
        angles: (theta, phi) coordinates
        original_neighbors: original k-step neighbors at current resolution
    """
    # Step 1: Find k-step neighbors at current resolution
    neighbor_pixels, step_levels = get_s2_k_step_neighbors(center_pix, k_steps, cur_res)

    # Step 2: Sample at higher resolution within this neighborhood
    high_res_pixels, angles = sample_s2_higher_resolution(
        neighbor_pixels, cur_res, target_res
    )

    return high_res_pixels, angles, neighbor_pixels


def get_s1_local_sampling(center_s1_idx, k_steps, cur_res, target_res):
    """
    Sample S1 indices at higher resolution within the k-step neighborhood

    Args:
        center_s1_idx: int, center S1 index at current resolution
        k_steps: int, maximum number of steps for neighborhood
        cur_res: int, current resolution level
        target_res: int, target higher resolution level

    Returns:
        high_res_s1_indices: array of S1 indices at target resolution
        angles: array of psi angles (in radians)
        original_neighbors: original k-step neighbors at current resolution
    """
    if target_res <= cur_res:
        raise ValueError("Target resolution must be higher than current resolution")

    # Get S1 grid size at current resolution
    Npix_s1_cur = 6 * 2**cur_res

    # Find k-step neighbors at current resolution using BFS
    visited = set()
    neighbors_by_step = {0: [center_s1_idx]}
    queue = [(center_s1_idx, 0)]
    visited.add(center_s1_idx)

    while queue:
        current_idx, current_step = queue.pop(0)

        if current_step < k_steps:
            # Get immediate neighbors (wrap around for circular S1)
            immediate_neighbors = [
                (current_idx - 1) % Npix_s1_cur,
                (current_idx + 1) % Npix_s1_cur,
            ]

            for neighbor_idx in immediate_neighbors:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    next_step = current_step + 1

                    if next_step not in neighbors_by_step:
                        neighbors_by_step[next_step] = []
                    neighbors_by_step[next_step].append(neighbor_idx)

                    queue.append((neighbor_idx, next_step))

    # Flatten all k-step neighbors
    all_neighbors = []
    for step in range(k_steps + 1):
        if step in neighbors_by_step:
            all_neighbors.extend(neighbors_by_step[step])

    neighbor_s1_indices = np.array(all_neighbors)

    # Sample at higher resolution using subdivision
    # Each S1 index at cur_res maps to multiple indices at target_res
    subdivision_factor = 2 ** (target_res - cur_res)

    high_res_s1_indices = []
    for s1_idx in neighbor_s1_indices:
        # Each S1 index subdivides into consecutive indices
        start_idx = s1_idx * subdivision_factor
        end_idx = start_idx + subdivision_factor
        high_res_s1_indices.extend(range(start_idx, end_idx))

    high_res_s1_indices = np.array(high_res_s1_indices)

    # Convert to angles (psi values)
    Npix_s1_target = 6 * 2**target_res
    dt = 2 * np.pi / Npix_s1_target
    psi_angles = high_res_s1_indices * dt + dt / 2  # Center of each bin

    return high_res_s1_indices, psi_angles, neighbor_s1_indices


def get_so3_neighbor_current_res(quat, s2_idx, s1_idx, cur_res, n_neighbors=8):
    """
    Fengyu:
        TODO: Maybe find a better metric instead of ||q1-q2||.

    Get SO(3) neighbors at the CURRENT resolution level using healpy
    Similar to get_neighbor but for current resolution instead of next resolution

    Args:
        quat: (4,) quaternion at current resolution
        s2_idx: int, S2 index at current resolution
        s1_idx: int, S1 index at current resolution
        cur_res: int, current resolution level
        n_neighbors: int, number of neighbors to return (default 8)

    Returns:
        neighbor_quats: (n_neighbors, 4) array of neighbor quaternions
        neighbor_indices: (n_neighbors, 2) array of [s2_idx, s1_idx] pairs
    """
    try:
        import healpy
    except ImportError:
        raise RuntimeError("You need to `pip install healpy` for neighbor finding.")

    Nside = 2**cur_res
    Npix_s1 = 6 * 2**cur_res

    # Get S2 neighbors at current resolution
    s2_neighbors = healpy.get_all_neighbours(Nside, s2_idx, nest=True)
    s2_neighbors = s2_neighbors[s2_neighbors != -1]  # Remove invalid neighbors
    s2_neighbors = np.append(s2_neighbors, s2_idx)  # Include center

    # Get extended S1 neighbors at current resolution (vectorized)
    s1_offsets = np.arange(-2, 3)  # [-2, -1, 0, 1, 2]
    s1_neighbors = (s1_idx + s1_offsets) % Npix_s1

    # Create all combinations using meshgrid (vectorized)
    s2_mesh, s1_mesh = np.meshgrid(s2_neighbors, s1_neighbors, indexing="ij")
    candidate_indices = np.column_stack([s2_mesh.ravel(), s1_mesh.ravel()])

    # Convert all candidates to quaternions (vectorized)
    s2_indices = candidate_indices[:, 0]
    s1_indices = candidate_indices[:, 1]

    # Get angles (vectorized)
    theta, phi = pix2ang(Nside, s2_indices, nest=True)
    dt = 2 * np.pi / Npix_s1
    psi = s1_indices * dt + dt / 2

    # Convert to quaternions (vectorized)
    candidate_quats = hopf_to_quat(theta, phi, psi)

    # Remove the center point (vectorized boolean indexing)
    center_mask = ~(
        (candidate_indices[:, 0] == s2_idx) & (candidate_indices[:, 1] == s1_idx)
    )
    candidate_quats = candidate_quats[center_mask]
    candidate_indices = candidate_indices[center_mask]

    if len(candidate_quats) == 0:
        return np.array([]), np.array([])

    # Calculate quaternion distances (vectorized)
    if quat.ndim > 1:
        quat = quat.flatten()[:4]  # Ensure it's 1D with 4 elements

    # Vectorized distance calculation with broadcasting
    diff_pos = candidate_quats - quat[None, :]  # Shape: (N, 4)
    diff_neg = candidate_quats + quat[None, :]  # Shape: (N, 4)

    dist_pos = np.sum(diff_pos**2, axis=1)  # Shape: (N,)
    dist_neg = np.sum(diff_neg**2, axis=1)  # Shape: (N,)

    dists = np.minimum(dist_pos, dist_neg)

    # Select the n_neighbors closest candidates (vectorized)
    n_to_select = min(n_neighbors, len(candidate_quats))
    closest_indices = np.argpartition(dists, n_to_select)[:n_to_select]

    # Sort the selected neighbors by distance
    closest_sorted = closest_indices[np.argsort(dists[closest_indices])]

    neighbor_quats = candidate_quats[closest_sorted]
    neighbor_indices = candidate_indices[closest_sorted]

    return neighbor_quats, neighbor_indices


def get_so3_k_step_neighbors(
    center_quat, center_s2_idx, center_s1_idx, k_steps, cur_res
):
    """
    Find k-step neighbors in SO(3) space using current resolution neighbors

    Args:
        center_quat: (4,) center quaternion
        center_s2_idx: int, center S2 index
        center_s1_idx: int, center S1 index
        k_steps: int, maximum number of steps
        cur_res: int, current resolution level

    Returns:
        neighbor_quats: (N, 4) array of neighbor quaternions
        neighbor_indices: (N, 2) array of [s2_idx, s1_idx] pairs
        step_levels: (N,) array of step distances
    """
    visited = set()
    neighbors_by_step = {0: [(center_quat, center_s2_idx, center_s1_idx)]}
    queue = [(center_quat, center_s2_idx, center_s1_idx, 0)]
    visited.add((center_s2_idx, center_s1_idx))

    while queue:
        current_quat, current_s2, current_s1, current_step = queue.pop(0)

        if current_step < k_steps:
            # Use the new function to get neighbors at current resolution
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

    # Flatten all neighbors
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
    """
    Sample SO(3) at higher resolution within the k-step neighborhood

    Args:
        neighbor_indices: (N, 2) array of [s2_idx, s1_idx] at current resolution
        cur_res: int, current resolution level
        target_res: int, target higher resolution level

    Returns:
        high_res_indices: (M, 2) array of [s2_idx, s1_idx] at target resolution
        high_res_quats: (M, 4) array of quaternions at target resolution
        high_res_rots: (M, 3, 3) tensor of rotation matrices
    """
    if target_res <= cur_res:
        raise ValueError("Target resolution must be higher than current resolution")

    # Calculate subdivision factors
    s2_subdivision_factor = 4 ** (target_res - cur_res)
    s1_subdivision_factor = 2 ** (target_res - cur_res)

    high_res_indices = []

    for s2_idx, s1_idx in neighbor_indices:
        # Subdivide S2 index
        s2_start = s2_idx * s2_subdivision_factor
        s2_end = s2_start + s2_subdivision_factor
        s2_children = list(range(s2_start, s2_end))

        # Subdivide S1 index
        s1_start = s1_idx * s1_subdivision_factor
        s1_end = s1_start + s1_subdivision_factor
        s1_children = list(range(s1_start, s1_end))

        # Create all combinations
        for s2_child in s2_children:
            for s1_child in s1_children:
                high_res_indices.append([s2_child, s1_child])

    high_res_indices = np.array(high_res_indices)

    # Convert to quaternions and rotation matrices
    s2_indices = high_res_indices[:, 0]
    s1_indices = high_res_indices[:, 1]

    # Get angles
    Nside_target = 2**target_res
    theta, phi = pix2ang(Nside_target, s2_indices, nest=True)

    Npix_s1_target = 6 * 2**target_res
    dt = 2 * np.pi / Npix_s1_target
    psi = s1_indices * dt + dt / 2

    # Convert to quaternions
    high_res_quats = hopf_to_quat(theta, phi, psi)

    # Convert to rotation matrices
    high_res_rots = quaternion_to_matrix(torch.from_numpy(high_res_quats))

    return high_res_indices, high_res_quats, high_res_rots, (theta, phi, psi)


def get_so3_local_sampling(
    center_quat, center_s2_idx, center_s1_idx, k_steps, cur_res, target_res
):
    """
    Complete SO(3) local sampling using proper SO(3) neighbors

    Args:
        center_quat: (4,) center quaternion
        center_s2_idx: int, center S2 index
        center_s1_idx: int, center S1 index
        k_steps: int, neighborhood radius in steps
        cur_res: int, current resolution level
        target_res: int, target sampling resolution

    Returns:
        high_res_indices: (N, 2) array of [s2_idx, s1_idx] pairs
        high_res_quats: (N, 4) array of quaternions
        high_res_rots: (N, 3, 3) tensor of rotation matrices
        original_neighbors: original k-step neighbors at current resolution
    """
    # Step 1: Find k-step neighbors in SO(3) space
    neighbor_quats, neighbor_indices, step_levels = get_so3_k_step_neighbors(
        center_quat, center_s2_idx, center_s1_idx, k_steps, cur_res
    )
    if target_res > cur_res:
        # Step 2: Sample at higher resolution within this neighborhood
        indices, quats, rots, angles = sample_so3_higher_resolution(
            neighbor_indices, cur_res, target_res
        )
    elif target_res == cur_res:
        indices = neighbor_indices
        quats = neighbor_quats
        rots = quaternion_to_matrix(torch.from_numpy(neighbor_quats))
        s2_indices = indices[:, 0]
        s1_indices = indices[:, 1]

        # Get angles
        Nside_target = 2**target_res
        theta, phi = pix2ang(Nside_target, s2_indices, nest=True)

        Npix_s1_target = 6 * 2**target_res
        dt = 2 * np.pi / Npix_s1_target
        psi = s1_indices * dt + dt / 2

        angles = (theta, phi, psi)
    else:
        raise ValueError("Current resolution cannot be higher than target resolution")

    return indices, quats, rots, angles


# Example usage function for your LocalPoseSearcher
def get_local_s2_poses(center_s2_pix, center_s1_idx, k_steps, cur_res, target_res):
    """
    Generate local pose samples around a center pose

    Args:
        center_s2_pix: int, center S2 pixel index
        center_s1_idx: int, center S1 index
        k_steps: int, neighborhood radius
        cur_res: int, current resolution
        target_res: int, target sampling resolution

    Returns:
        pose_samples: array of (s2_pix, s1_idx) pairs
        quaternions: corresponding quaternions
    """
    # Get high-resolution S2 samples
    s2_high_res_pixels, (theta, phi), _ = get_s2_local_sampling(
        center_s2_pix, k_steps, cur_res, target_res
    )

    # Get high-resolution S1 samples
    s1_high_res_indices, psi_angles, _ = get_s1_local_sampling(
        center_s1_idx, k_steps, cur_res, target_res
    )

    # Create all combinations of S2 and S1 samples
    pose_samples = []
    all_theta = []
    all_phi = []
    all_psi = []

    for s2_pix, t, p in zip(s2_high_res_pixels, theta, phi):
        for s1_idx, psi in zip(s1_high_res_indices, psi_angles):
            pose_samples.append([s2_pix, s1_idx])
            all_theta.append(t)
            all_phi.append(p)
            all_psi.append(psi)

    pose_samples = np.array(pose_samples)
    all_theta = np.array(all_theta)
    all_phi = np.array(all_phi)
    all_psi = np.array(all_psi)

    # Convert to quaternions
    quaternions = hopf_to_quat(all_theta, all_phi, all_psi)

    return pose_samples, quaternions


def get_quat_from_ind(resol, s2i, s1i):
    """
    Convert SO(3) grid indices to Euler angles and quaternions at given resolution

    Args:
        resol: int, resolution level
        s2i: int or array, S2 (HEALPix) index
        s1i: int or array, S1 (circle) index

    Returns:
        euler_angles: (N, 3) array of (phi, theta, psi) Euler angles in radians
        quaternions: (N, 4) array of quaternions
    """
    # Handle single indices or arrays
    s2i = np.asarray(s2i)
    s1i = np.asarray(s1i)

    # Ensure they're the same shape
    if s2i.shape != s1i.shape:
        raise ValueError("s2i and s1i must have the same shape")

    original_shape = s2i.shape
    s2i_flat = s2i.flatten()
    s1i_flat = s1i.flatten()

    # Get S2 angles (theta, phi) from HEALPix indices
    Nside = 2**resol
    theta, phi = pix2ang(Nside, s2i_flat, nest=True)

    # Get S1 angles (psi) from S1 indices
    Npix_s1 = 6 * 2**resol
    dt = 2 * np.pi / Npix_s1
    psi = s1i_flat * dt + dt / 2

    # Convert to quaternions using Hopf coordinates
    quaternions = hopf_to_quat(theta, phi, psi)

    # Normalize quaternions
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)

    # Reshape back to original shape if needed
    if original_shape:
        quaternions = quaternions.reshape(*original_shape, 4)

    return quaternions


def euler_local_sampling(eulers, target_res, k_steps=1):
    """
    Local search around given Euler angles using angular steps from target_res grid

    Args:
        eulers: torch.Tensor (3,) or (N, 3) Euler angles [φ, θ, ψ] to search around
        target_res: int, target resolution for angular step size
        k_steps: int, number of steps to search in each direction (default 1)

    Returns:
        candidate_eulers: torch.Tensor (M, 3) of candidate Euler angles [φ, θ, ψ]
        candidate_quats: torch.Tensor (M, 4) of candidate quaternions
        candidate_rots: torch.Tensor (M, 3, 3) of rotation matrices

    Note:
        Angular steps are determined by target_res:
        - S2 (theta, phi): based on HEALPix resolution
        - S1 (psi): 2π / (6 * 2^target_res)
    """
    # Convert to torch if needed
    if isinstance(eulers, np.ndarray):
        eulers = torch.from_numpy(eulers.astype(np.float32))

    if eulers.ndim == 1:
        eulers = eulers.unsqueeze(0)

    # Use input Euler angles directly
    euler_angles = eulers  # (N, 3) [φ, θ, ψ]

    # Calculate angular steps for target resolution
    Nside_target = 2**target_res
    theta_step = np.sqrt(4 * np.pi / (12 * Nside_target**2))
    phi_step = theta_step

    Npix_s1_target = 6 * 2**target_res
    psi_step = 2 * np.pi / Npix_s1_target

    # Generate perturbation offsets (on CPU, then move to device)
    k_range = torch.arange(-k_steps, k_steps + 1, dtype=torch.float32)
    k_phi_grid, k_theta_grid, k_psi_grid = torch.meshgrid(
        k_range, k_range, k_range, indexing="ij"
    )
    k_phi_flat = k_phi_grid.reshape(-1)  # Shape: (n_offsets,)
    k_theta_flat = k_theta_grid.reshape(-1)
    k_psi_flat = k_psi_grid.reshape(-1)

    # Move to same device as eulers
    device = eulers.device
    k_phi_flat = k_phi_flat.to(device)
    k_theta_flat = k_theta_flat.to(device)
    k_psi_flat = k_psi_flat.to(device)

    # Generate candidates for all input quaternions at once
    # euler_angles: (N, 3), k_flat: (n_offsets,)
    # We want: (N, n_offsets, 3)
    n_inputs = euler_angles.shape[0]
    n_offsets = k_phi_flat.shape[0]

    # Expand dimensions for broadcasting
    phi_center = euler_angles[:, 0].unsqueeze(1)  # (N, 1)
    theta_center = euler_angles[:, 1].unsqueeze(1)
    psi_center = euler_angles[:, 2].unsqueeze(1)

    # Perturb angles (vectorized over all inputs)
    phi_candidates = phi_center + k_phi_flat.unsqueeze(0) * phi_step  # (N, n_offsets)
    theta_candidates = theta_center + k_theta_flat.unsqueeze(0) * theta_step
    psi_candidates = psi_center + k_psi_flat.unsqueeze(0) * psi_step

    # Wrap phi to [-π, π] (vectorized)
    phi_candidates = phi_candidates % (2 * np.pi)
    phi_candidates = torch.where(
        phi_candidates > np.pi, phi_candidates - 2 * np.pi, phi_candidates
    )

    # Wrap psi to [-π, π] (vectorized)
    psi_candidates = psi_candidates % (2 * np.pi)
    psi_candidates = torch.where(
        psi_candidates > np.pi, psi_candidates - 2 * np.pi, psi_candidates
    )

    # Clamp theta to [0, π] (vectorized)
    theta_candidates = torch.clamp(theta_candidates, 0, np.pi)

    # Stack and flatten: (N, n_offsets, 3) -> (N*n_offsets, 3)
    candidate_eulers = torch.stack(
        [phi_candidates, theta_candidates, psi_candidates], dim=2
    )
    candidate_eulers = candidate_eulers.reshape(-1, 3)

    # Convert Euler angles to rotation matrices using euler_to_matrix
    candidate_rots = euler_to_matrix(candidate_eulers, device=device)

    # Convert rotation matrices to quaternions
    candidate_quats = matrix_to_quaternion(candidate_rots)

    return candidate_eulers, candidate_quats, candidate_rots