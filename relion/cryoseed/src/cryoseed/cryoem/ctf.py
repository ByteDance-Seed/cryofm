from __future__ import annotations

import torch
from torch import Tensor

from cryoseed.fft.coords import fftfreq_coords2d

__all__ = ["compute_ctf", "ctf_from_params"]


def compute_ctf(
    side_length: int,
    angpix: float | Tensor,
    kV: Tensor,
    Cs: Tensor,
    Bfac: Tensor | None,
    scale: Tensor | None,
    Q0: Tensor,
    phase_shift: Tensor | None,
    DeltafU: Tensor,
    DeltafV: Tensor,
    azimuthal_angle: Tensor,
    device=None,
) -> Tensor:
    """Compute CTF for the full image.

    Relion source: ctf.cpp/CTF::initialise, ctf.cpp/CTF::getFftwImage, ctf.h/getCTF

    Returns:
        CTF tensor with shape ``(N, L, L)``.
    """

    side_length = int(side_length)
    if side_length <= 0:
        raise ValueError(f"side_length must be > 0, got {side_length}")

    if device is None:
        device = DeltafU.device

    dtype = torch.float32
    for t in (kV, Cs, Bfac, scale, Q0, phase_shift, DeltafU, DeltafV, azimuthal_angle):
        if t is None:
            continue
        if isinstance(t, Tensor) and t.dtype.is_floating_point:
            dtype = torch.promote_types(dtype, t.dtype)

    def _as_float(x: float | Tensor) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(x, device=device, dtype=dtype)

    angpix_t = _as_float(angpix)
    kV = _as_float(kV)
    Cs = _as_float(Cs)
    Bfac = _as_float(0.0 if Bfac is None else Bfac)
    scale = _as_float(1.0 if scale is None else scale)
    Q0 = _as_float(Q0)
    phase_shift = _as_float(0.0 if phase_shift is None else phase_shift)
    DeltafU = _as_float(DeltafU)
    DeltafV = _as_float(DeltafV)
    azimuthal_angle = _as_float(azimuthal_angle)

    local_Cs = Cs * 1e7
    local_kV = kV * 1e3
    rad_azimuth = torch.deg2rad(azimuthal_angle)

    lambda_A = 12.2643247 / torch.sqrt(local_kV * (1 + local_kV * 0.978466e-6))

    PI = torch.pi
    k_defocus = PI * lambda_A
    k_spherical = 0.5 * PI * local_Cs * lambda_A**3
    phase_contrast = torch.atan(Q0 / torch.sqrt(1 - Q0**2))
    b_envelope = -Bfac / 4.0
    phase_shift_rad = torch.deg2rad(phase_shift)

    cos_az = torch.cos(rad_azimuth)
    sin_az = torch.sin(rad_azimuth)

    Q = torch.stack(
        [
            torch.stack([cos_az, sin_az], dim=-1),
            torch.stack([-sin_az, cos_az], dim=-1),
        ],
        dim=1,
    )

    D = torch.zeros((DeltafU.shape[0], 2, 2), device=device, dtype=dtype)
    D[:, 0, 0] = -DeltafU
    D[:, 1, 1] = -DeltafV

    A = torch.bmm(Q.transpose(1, 2), torch.bmm(D, Q))

    defocus_xx = A[:, 0, 0][:, None, None]
    defocus_xy = A[:, 0, 1][:, None, None]
    defocus_yy = A[:, 1, 1][:, None, None]

    k_defocus = k_defocus[:, None, None]
    k_spherical = k_spherical[:, None, None]
    phase_contrast = phase_contrast[:, None, None]
    b_envelope = b_envelope[:, None, None]
    phase_shift_rad = phase_shift_rad[:, None, None]
    scale = scale[:, None, None]

    freq_coords = fftfreq_coords2d(side_length, device=device, dtype=dtype) / angpix_t
    freq_x = freq_coords[:, 0].view(side_length, side_length)[None, :, :]
    freq_y = freq_coords[:, 1].view(side_length, side_length)[None, :, :]

    freq_sq = freq_x**2 + freq_y**2

    phase = (
        k_defocus
        * (
            defocus_xx * freq_x**2
            + 2 * defocus_xy * freq_x * freq_y
            + defocus_yy * freq_y**2
        )
        + k_spherical * freq_sq**2
        - phase_shift_rad
        - phase_contrast
    )

    ctf = -torch.sin(phase)
    ctf = ctf * torch.exp(b_envelope * freq_sq)
    ctf = ctf * scale

    return torch.where(torch.abs(ctf) < 1e-8, torch.sign(ctf) * 1e-8, ctf)


def ctf_from_params(
    ctf_params: Tensor | dict,
    side_length: int,
    angpix: float | Tensor | None = None,
    device=None,
) -> Tensor:
    """Compute CTF from a packed per-particle parameter representation.

    The primary convention matches :meth:`cryoseed.data.ParticleDataset._get_ctf_params_tensor`:

    ``[kV, Cs, Bfac, scale, Q0, phase_shift, DeltafU, DeltafV, azimuthal_angle]``.

    Args:
        ctf_params:
            Either a tensor of shape ``(B, 9)`` (or ``(9,)`` for a single particle),
            or a dict containing the same fields.

            If a tensor with shape ``(B, 10)`` (or ``(10,)``) is provided and
            ``angpix`` is not passed, the first entry is interpreted as ``angpix``
            and the remaining 9 entries follow the convention above.
        side_length:
            Image size ``L``. The returned tensor has shape ``(B, L, L)``.
        angpix:
            Pixel size in Angstroms/pixel. Required unless it is embedded in
            ``ctf_params`` as described above.
        device:
            Optional device override.

    Returns:
        CTF tensor with shape ``(B, L, L)``.
    """

    if isinstance(ctf_params, dict):
        if angpix is None:
            angpix = ctf_params.get("angpix", None)
        if angpix is None:
            raise ValueError("angpix must be provided (either as an argument or in ctf_params['angpix']")

        return compute_ctf(
            side_length=side_length,
            angpix=angpix,
            kV=ctf_params["kV"],
            Cs=ctf_params["Cs"],
            Bfac=ctf_params["Bfac"],
            scale=ctf_params["scale"],
            Q0=ctf_params["Q0"],
            phase_shift=ctf_params["phase_shift"],
            DeltafU=ctf_params["DeltafU"],
            DeltafV=ctf_params["DeltafV"],
            azimuthal_angle=ctf_params["azimuthal_angle"],
            device=device,
        )

    params = ctf_params
    if params.ndim == 1:
        params = params[None, :]
    if params.ndim != 2:
        raise ValueError(
            f"ctf_params must have shape (B,9)/(B,10) or (9,)/(10,), got {tuple(ctf_params.shape)}"
        )

    if params.shape[1] == 10 and angpix is None:
        angpix = params[:, 0]
        params = params[:, 1:]

    if params.shape[1] != 9:
        raise ValueError(f"ctf_params last dim must be 9 (or 10 with embedded angpix), got {params.shape[1]}")

    if angpix is None:
        raise ValueError("angpix must be provided (either as an argument or embedded in ctf_params)")

    kV, Cs, Bfac, scale, Q0, phase_shift, DeltafU, DeltafV, azimuthal_angle = params.unbind(dim=1)
    return compute_ctf(
        side_length=side_length,
        angpix=angpix,
        kV=kV,
        Cs=Cs,
        Bfac=Bfac,
        scale=scale,
        Q0=Q0,
        phase_shift=phase_shift,
        DeltafU=DeltafU,
        DeltafV=DeltafV,
        azimuthal_angle=azimuthal_angle,
        device=device,
    )