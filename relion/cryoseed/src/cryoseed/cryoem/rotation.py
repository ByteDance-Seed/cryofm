from __future__ import annotations

"""Rotation conventions used throughout cryoSeed.

All angles are in radians.

**Euler angles (ZYZ)**

We use a ZYZ Euler convention with angles ordered as ``[rot, tilt, psi]`` and
rotation matrices defined by

    ``R = Rz(rot) @ Ry(tilt) @ Rz(psi)``.

Unless stated otherwise, Euler angles returned by this module are
canonicalized to:

- ``rot`` and ``psi`` in ``[-pi, pi)``
- ``tilt`` in ``[0, pi]``

**How rotation matrices act on points**

Across cryoSeed backends (projection / backprojection), 3D coordinates are
treated as row-vectors in ``(x, y, z)`` order and are right-multiplied:

    ``p_rot = p @ R``.

If you prefer column-vector notation, the same transformation is

    ``p_rot_col = R.T @ p_col``.

So the numeric 3×3 matrix ``R`` is the same; only the vector representation
(row vs column) changes where the transpose appears.

**"Rotate the volume" vs "rotate the sampling coordinates"**

In imaging operators we typically interpret ``R`` as the rotation of the
*volume* relative to the detector/world frame. The equivalent sampling
operation is implemented by transforming the sampling coordinates in the
opposite sense (relative motion). In the row-vector convention above this is
expressed as ``coords = coords @ R``.

**RELION interop**

RELION defines Euler angles ``[rot, tilt, psi]`` (radians) in a column-vector
convention with

    ``R_relion = Rz(rot) @ Ry(tilt) @ Rz(psi)``.

To use the corresponding cryoSeed row-vector convention, we use

    ``R_cryoseed = R_relion.T``.

This is why :func:`relion_euler_to_matrix` returns ``euler_to_matrix(...,
transpose=True)`` and :func:`from_relion_euler` returns Euler angles whose
matrix matches ``R_relion.T``.
"""

import torch
from torch import Tensor

__all__ = [
    "from_relion_euler",
    "euler_to_matrix",
    "relion_euler_to_matrix",
    "matrix_to_euler",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "euler_zyz_to_quaternion",
    "quaternion_to_zyz_euler",
    "relative_rotation_error",
]


def _wrap_to_pi(x: Tensor) -> Tensor:
    r"""Wrap angles to ``[-pi, pi)``."""
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def from_relion_euler(relion_euler: Tensor) -> Tensor:
    r"""Convert RELION Euler angles to canonical cryoSeed ZYZ Euler angles.

    RELION angles are ordered as ``[rot, tilt, psi]`` and define the column-vector
    rotation

        ``R_relion = Rz(rot) @ Ry(tilt) @ Rz(psi)``

    cryoSeed internally uses row-vectors with right multiplication, so the
    corresponding internal rotation matrix is

        ``R_cryoseed = R_relion.T``

    We define cryoSeed Euler angles, also ordered as ``[rot, tilt, psi]``, by

        ``R_cryoseed = Rz(rot) @ Ry(tilt) @ Rz(psi)``

    and return a canonical representative with:
    - ``rot, psi`` in ``[-pi, pi)``
    - ``tilt`` in ``[0, pi]``

    Args:
        relion_euler: Tensor of shape ``(..., 3)`` ordered as
            ``[rot, tilt, psi]`` in radians.

    Returns:
        Tensor of shape ``(..., 3)`` containing canonical cryoSeed Euler angles
        ordered as ``[rot, tilt, psi]``.
    """
    if relion_euler.shape[-1] != 3:
        raise ValueError(
            f"relion_euler must have shape (..., 3), got {tuple(relion_euler.shape)}"
        )

    relion_rot = relion_euler[..., 0]
    relion_tilt = relion_euler[..., 1]
    relion_psi = relion_euler[..., 2]

    cryoseed_rot = _wrap_to_pi(torch.pi - relion_psi)
    cryoseed_tilt = relion_tilt
    cryoseed_psi = _wrap_to_pi(torch.pi - relion_rot)

    return torch.stack([cryoseed_rot, cryoseed_tilt, cryoseed_psi], dim=-1)


def euler_to_matrix(
    euler: Tensor,
    *,
    transpose: bool = False,
    device=None,
    dtype=None,
) -> Tensor:
    r"""Convert ZYZ Euler angles to rotation matrices.

    The input is ordered as ``[rot, tilt, psi]`` and defines

        ``R = Rz(rot) @ Ry(tilt) @ Rz(psi)``

    This is the internal cryoSeed ZYZ convention.

    Args:
        euler: Euler angles in radians with shape ``(..., 3)``, ordered as
            ``[rot, tilt, psi]``.
        transpose: If ``True``, return ``R.T``. Otherwise return ``R``.

    Returns:
        Rotation matrices with shape ``(..., 3, 3)``.
    """
    if euler.shape[-1] != 3:
        raise ValueError(f"euler must have shape (..., 3), got {tuple(euler.shape)}")

    if dtype is None and not torch.is_floating_point(euler):
        dtype = torch.float32

    if device is not None or dtype is not None:
        euler = euler.to(
            device=euler.device if device is None else device,
            dtype=euler.dtype if dtype is None else dtype,
        )

    rot = euler[..., 0]
    tilt = euler[..., 1]
    psi = euler[..., 2]

    cr, sr = torch.cos(rot), torch.sin(rot)
    ct, st = torch.cos(tilt), torch.sin(tilt)
    cp, sp = torch.cos(psi), torch.sin(psi)

    zeros = torch.zeros_like(rot)
    ones = torch.ones_like(rot)

    rz_rot = torch.stack(
        [
            torch.stack([cr, -sr, zeros], dim=-1),
            torch.stack([sr, cr, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    ry_tilt = torch.stack(
        [
            torch.stack([ct, zeros, st], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-st, zeros, ct], dim=-1),
        ],
        dim=-2,
    )

    rz_psi = torch.stack(
        [
            torch.stack([cp, -sp, zeros], dim=-1),
            torch.stack([sp, cp, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    matrix = rz_rot @ ry_tilt @ rz_psi
    return matrix.transpose(-1, -2) if transpose else matrix


def relion_euler_to_matrix(
    relion_euler: Tensor,
    *,
    device=None,
    dtype=None,
) -> Tensor:
    r"""Convert RELION Euler angles directly to cryoSeed rotation matrices.

    RELION angles are ordered as ``[rot, tilt, psi]`` and define

        ``R_relion = Rz(rot) @ Ry(tilt) @ Rz(psi)``

    in column-vector convention. cryoSeed uses the corresponding row-vector
    matrix

        ``R_cryoseed = R_relion.T``

    Args:
        relion_euler: Tensor of shape ``(..., 3)`` ordered as
            ``[rot, tilt, psi]`` in radians.

    Returns:
        Rotation matrices with shape ``(..., 3, 3)`` in cryoSeed convention.
    """
    return euler_to_matrix(relion_euler, transpose=True, device=device, dtype=dtype)


def matrix_to_euler(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Convert rotation matrices to canonical cryoSeed ZYZ Euler angles.

    Returns Euler angles ``[rot, tilt, psi]`` such that

        ``R = Rz(rot) @ Ry(tilt) @ Rz(psi)``

    This is the inverse convention of::

        euler_to_matrix(euler, transpose=False)

    The returned angles are canonicalized to:
    - ``rot, psi`` in ``[-pi, pi)``
    - ``tilt`` in ``[0, pi]``

    Args:
        R: Rotation matrices of shape ``(3, 3)`` or ``(..., 3, 3)``.
        eps: Threshold for detecting gimbal lock when ``sin(tilt)`` is near zero.

    Returns:
        Euler angles of shape ``(..., 3)`` ordered as ``[rot, tilt, psi]``.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (..., 3, 3), got {tuple(R.shape)}")

    single = R.ndim == 2
    if single:
        R = R.unsqueeze(0)

    r00 = R[..., 0, 0]
    r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]
    r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    tilt = torch.acos(torch.clamp(r22, -1.0, 1.0))
    sin_tilt = torch.sin(tilt)

    gimbal_lock = torch.abs(sin_tilt) < eps
    normal = ~gimbal_lock

    rot = torch.zeros_like(tilt)
    psi = torch.zeros_like(tilt)

    # Normal case:
    # r02 = cos(rot) sin(tilt), r12 = sin(rot) sin(tilt)
    # r20 = -sin(tilt) cos(psi), r21 = sin(tilt) sin(psi)
    rot[normal] = torch.atan2(r12[normal], r02[normal])
    psi[normal] = torch.atan2(r21[normal], -r20[normal])

    if torch.any(gimbal_lock):
        # tilt ~= 0: R = Rz(rot + psi)
        mask0 = gimbal_lock & (r22 > 0)
        rot[mask0] = torch.atan2(r10[mask0], r00[mask0])
        psi[mask0] = 0.0

        # tilt ~= pi: choose canonical psi = 0
        mask_pi = gimbal_lock & ~mask0
        rot[mask_pi] = torch.atan2(-r10[mask_pi], -r00[mask_pi])
        psi[mask_pi] = 0.0

    rot = _wrap_to_pi(rot)
    psi = _wrap_to_pi(psi)

    euler = torch.stack([rot, tilt, psi], dim=-1)
    if single:
        euler = euler.squeeze(0)

    return euler.to(dtype=R.dtype)


def quaternion_to_matrix(
    quat: Tensor,
    *,
    scalar_first: bool = True,
    normalize: bool = True,
) -> Tensor:
    r"""Convert quaternions to rotation matrices.

    The quaternion is interpreted as:
    - ``[w, x, y, z]`` if ``scalar_first=True``
    - ``[x, y, z, w]`` if ``scalar_first=False``

    The returned rotation matrix ``R`` follows the same convention as
    :func:`euler_to_matrix`, i.e. it is the literal 3x3 rotation matrix.
    How it acts on vectors depends on whether your vectors are represented as
    columns or rows, but the matrix itself is the same.

    Args:
        quat: Quaternion tensor/array of shape ``(..., 4)``.
        scalar_first: Whether the quaternion is stored as ``[w, x, y, z]``.
        normalize: If ``True``, normalize the input quaternion before conversion.

    Returns:
        Rotation matrices of shape ``(..., 3, 3)``.
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"quat must have shape (..., 4), got {tuple(quat.shape)}")

    if not torch.is_floating_point(quat):
        quat = quat.to(torch.float32)

    if scalar_first:
        w, x, y, z = quat.unbind(dim=-1)
    else:
        x, y, z, w = quat.unbind(dim=-1)

    if normalize:
        norm = torch.sqrt(w * w + x * x + y * y + z * z).clamp_min(1e-12)
        w = w / norm
        x = x / norm
        y = y / norm
        z = z / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_quaternion(
    R: Tensor,
    *,
    scalar_first: bool = True,
    canonical: bool = True,
) -> Tensor:
    r"""Convert rotation matrices to quaternions.

    Args:
        R: Rotation matrices of shape ``(3, 3)`` or ``(..., 3, 3)``.
        scalar_first: Whether to return ``[w, x, y, z]`` or ``[x, y, z, w]``.
        canonical: If ``True``, enforce a canonical sign by making the scalar
            part non-negative. This removes the ``q`` vs ``-q`` ambiguity.

    Returns:
        Quaternions of shape ``(..., 4)``.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (..., 3, 3), got {tuple(R.shape)}")

    if not torch.is_floating_point(R):
        R = R.to(torch.float32)

    single = R.ndim == 2
    if single:
        R = R.unsqueeze(0)

    m00 = R[..., 0, 0]
    m01 = R[..., 0, 1]
    m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]
    m11 = R[..., 1, 1]
    m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]
    m21 = R[..., 2, 1]
    m22 = R[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )

    qw = 0.5 * q_abs[..., 0]
    qx = 0.5 * q_abs[..., 1]
    qy = 0.5 * q_abs[..., 2]
    qz = 0.5 * q_abs[..., 3]

    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)

    quat = torch.stack([qw, qx, qy, qz], dim=-1)

    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)

    if canonical:
        sign = torch.where(quat[..., :1] < 0, -1.0, 1.0).to(quat.dtype)
        quat = quat * sign

    if not scalar_first:
        quat = quat[..., [1, 2, 3, 0]]

    if single:
        quat = quat.squeeze(0)

    return quat


def euler_zyz_to_quaternion(
    euler: Tensor,
    *,
    scalar_first: bool = True,
) -> Tensor:
    r"""Convert cryoSeed ZYZ Euler angles to quaternions."""
    R = euler_to_matrix(euler)
    return matrix_to_quaternion(R, scalar_first=scalar_first)


def quaternion_to_zyz_euler(
    quat: Tensor,
    *,
    scalar_first: bool = True,
) -> Tensor:
    r"""Convert quaternions to canonical cryoSeed ZYZ Euler angles."""
    R = quaternion_to_matrix(quat, scalar_first=scalar_first)
    return matrix_to_euler(R)


def relative_rotation_error(R1: Tensor, R2: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute the geodesic (relative) rotation error between two rotations.

    The error is defined as ``theta = acos((tr(R1^T R2) - 1) / 2)``.

    Args:
        R1: Rotation matrices of shape ``(..., 3, 3)``.
        R2: Rotation matrices of shape ``(..., 3, 3)``.
        eps: Small clamp value for numerical stability.

    Returns:
        Rotation error in radians with shape ``(...)``.
    """
    R_rel = R1.transpose(-1, -2) @ R2
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos_theta)