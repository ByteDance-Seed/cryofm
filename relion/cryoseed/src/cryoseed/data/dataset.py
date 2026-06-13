from __future__ import annotations

"""Dataset definitions for CryoSeed.

The primary entry point is :class:`~cryoseed.data.ParticleDataset`, a
:class:`torch.utils.data.Dataset` that reads RELION-style STAR metadata and
lazily memory-maps particle images from MRC/MRCS stacks.

Each sample is returned as a Python ``dict`` (rather than a single tensor) so
that pose and CTF metadata can travel alongside the image. Batching is handled
by :func:`cryoseed.data.data_collate_fn`.
"""

import os

import numpy as np
import torch
import mrcfile

from torch.utils.data import Dataset, get_worker_info

from .star import (
    read_starfile,
    parse_optics_parameters,
    parse_stack_entries,
    merge_optics_to_particles,
)

from cryoseed.cryoem.mask import circular_mask
from cryoseed.cryoem.rotation import from_relion_euler
from cryoseed.fft.fft_torch import fourier_to_primal_2d, primal_to_fourier_2d
from cryoseed.ops.transforms import downsample2d

__all__ = ["ParticleDataset"]


class ParticleDataset(Dataset):
    """Cryo-EM particle dataset backed by RELION STAR metadata.

    The dataset is defined by a RELION-style STAR file. Each item is loaded
    lazily from an MRC/MRCS stack and returned as a dictionary compatible with
    :func:`cryoseed.data.data_collate_fn`.

    Each sample dict contains (at least) the following keys:

    - ``"image_real"`` (torch.Tensor): Real-space particle image, shape ``(D, D)``,
      dtype ``float32``.
    - ``"particle_index"`` (int): Index in this dataset.
    - ``"stack_index"`` (int): Index inside the underlying MRC/MRCS stack.
    - ``"euler"`` (torch.Tensor | None): Euler angles in radians, shape ``(3,)``.
    - ``"trans"`` (torch.Tensor | None): In-plane translation in pixels, shape
      ``(2,)``.
    - ``"ctf_params"`` (torch.Tensor): CTF parameter vector, shape ``(9,)``.
    - ``"angpix"`` (float): Effective pixel size (Angstroms/pixel) after any
      downsampling.

    Args:
        star_path (str): Path to the input STAR file.
        data_prefix (str): Prefix prepended to relative MRC/MRCS paths parsed from
            the STAR file.
        num_particles (int | None): Optional cap on the dataset length. If set
            and smaller than the total number of particles, a random subset is
            selected.
        image_size (int | None): Target side length in pixels. If smaller than
            the full image size, particles are Fourier-downsampled.
        angpix (float | None): Pixel size override in Angstroms/pixel. If not
            provided, it is inferred from ``rlnImagePixelSize`` in the optics
            table.
        default_optic_params (dict | None): Fallback values for missing
            optics-level CTF fields (e.g. voltage and spherical aberration).
        default_particle_params (dict | None): Fallback values for missing
            particle-level CTF fields (e.g. defocus).

    Notes:
        - Translations are derived from ``rlnOriginXAngst``/``rlnOriginYAngst``
          when present and returned in pixels.
        - When downsampling, the image is masked with a circular support in
          Fourier space.
    """

    def __init__(
        self,
        star_path: str,
        data_prefix: str = "",
        num_particles: int | None = None,
        image_size: int | None = None,
        angpix: float | None = None,
        default_optic_params: dict | None = None,
        default_particle_params: dict | None = None,
    ):
        self.df_particles, self.df_optics = read_starfile(star_path)
        self.df_particles = merge_optics_to_particles(self.df_particles, self.df_optics)

        n_total = len(self.df_particles)
        if num_particles is not None:
            n_req = int(num_particles)
            if n_req > 0:
                n_keep = min(n_req, n_total)
                if n_keep < n_total:
                    rng = np.random.default_rng()
                    selected = np.sort(rng.choice(n_total, size=n_keep, replace=False))
                    self.df_particles = self.df_particles.iloc[selected].reset_index(drop=True)

        parsed_angpix, parsed_full_image_size = parse_optics_parameters(self.df_optics)

        if angpix is not None and float(angpix) > 0:
            self.angpix = float(angpix)
        elif parsed_angpix is not None:
            self.angpix = float(parsed_angpix)
        else:
            raise ValueError(
                "Missing pixel size (rlnImagePixelSize) in STAR optics table. "
                "Please pass angpix via CLI/config."
            )

        stack_index, stack_path = parse_stack_entries(
            self.df_particles,
            data_prefix,
        )

        if parsed_full_image_size is not None and int(parsed_full_image_size) > 0:
            self.full_image_size = int(parsed_full_image_size)
        else:
            if len(stack_path) == 0:
                raise ValueError("Cannot infer image size from an empty dataset")
            first_path = str(stack_path.iloc[0])
            with mrcfile.mmap(first_path, permissive=True) as mrcs:
                if mrcs.data is None:
                    raise ValueError(
                        f"MRC/MRCS file '{first_path}' has no readable data (mrcfile returned data=None). "
                        "The file may be truncated/corrupted, not an MRC/MRCS stack, or contains an invalid header."
                    )
                shape = tuple(mrcs.data.shape)
            if len(shape) < 2:
                raise ValueError(f"Invalid MRC data shape {shape} for '{first_path}'")
            if int(shape[-1]) != int(shape[-2]):
                raise ValueError(
                    f"Expected square particle images, but got shape {shape[-2:]} for '{first_path}'"
                )
            self.full_image_size = int(shape[-1])

        if image_size is not None and int(image_size) > 0:
            self.image_size = int(image_size)
        else:
            self.image_size = int(self.full_image_size)

        euler_cols = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
        if all(col in self.df_particles for col in euler_cols):
            self.euler_rad = np.deg2rad(
                self.df_particles[euler_cols].to_numpy(np.float32)
            ).astype(np.float32)
        else:
            self.euler_rad = None

        if "rlnOriginXAngst" in self.df_particles and "rlnOriginYAngst" in self.df_particles:
            x_A = self.df_particles["rlnOriginXAngst"].to_numpy(np.float32)
            y_A = self.df_particles["rlnOriginYAngst"].to_numpy(np.float32)
            self.shift_A = np.stack([x_A, y_A], axis=1).astype(np.float32)
        else:
            self.shift_A = None

        self.samples = list(zip(stack_path.tolist(), stack_index.tolist()))
        self._mrc_handles_by_worker: dict[tuple[int, str], object] = {}

        self.num_particles = len(self.samples)

        self.default_optic_params = dict(default_optic_params or {})
        self.default_particle_params = dict(default_particle_params or {})

        self._optic_fallback_params = {
            "kV": None,
            "Cs": None,
            "Bfac": 0.0,
            "scale": 1.0,
            "Q0": 0.1,
            "phase_shift": 0.0,
        }
        self._particle_fallback_params = {
            "DeltafU": None,
            "DeltafV": None,
            "azimuthal_angle": 90.0,
        }

        self._load_ctf_params()

    def __len__(self) -> int:
        """Return the number of particles in the dataset."""
        return len(self.samples)

    def effective_angpix(self) -> float:
        """Return the effective pixel size after any downsampling.

        If ``image_size == full_image_size`` this is equal to ``self.angpix``.
        Otherwise it is scaled by ``full_image_size / image_size``.

        Returns:
            float: Pixel size in Angstroms/pixel at the resolution of
            ``image_size``.

        Raises:
            ValueError: If ``image_size`` or ``full_image_size`` is not positive,
                or if ``image_size`` is larger than ``full_image_size``.
        """
        if int(self.image_size) <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")
        if int(self.full_image_size) <= 0:
            raise ValueError(f"full_image_size must be > 0, got {self.full_image_size}")

        if int(self.image_size) == int(self.full_image_size):
            return float(self.angpix)

        if int(self.image_size) > int(self.full_image_size):
            raise ValueError(
                f"image_size ({int(self.image_size)}) must be <= full_image_size ({int(self.full_image_size)})"
            )

        return float(self.angpix) * (float(self.full_image_size) / float(self.image_size))

    def _get_mrc_handle(self, path: str):
        """Return a worker-local memory-mapped handle for an MRC/MRCS file.

        This method caches handles keyed by ``(worker_id, path)`` so that a
        :class:`torch.utils.data.DataLoader` with multiple workers does not
        repeatedly reopen the same stack file.

        Args:
            path (str): Path to an MRC/MRCS file.

        Returns:
            A memory-mapped MRC handle as returned by :func:`mrcfile.mmap`.

        Raises:
            RuntimeError: If the file cannot be opened or does not contain
                readable image data.
        """
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else -1
        key = (worker_id, path)

        if key not in self._mrc_handles_by_worker:
            try:
                handle = mrcfile.mmap(path, permissive=True)
            except Exception as e:
                raise RuntimeError(f"Failed to open MRC/MRCS file '{path}'") from e

            if handle.data is None:
                try:
                    size_bytes = os.path.getsize(path)
                except OSError:
                    size_bytes = None

                try:
                    handle.close()
                except Exception:
                    pass

                msg = (
                    f"MRC/MRCS file '{path}' has no readable data (mrcfile returned data=None)."
                )
                if size_bytes is not None:
                    msg += f" File size: {size_bytes} bytes."
                msg += (
                    " The file may be truncated/corrupted, not an MRC/MRCS stack, "
                    "or contains an invalid header."
                )
                raise RuntimeError(msg)

            self._mrc_handles_by_worker[key] = handle

        return self._mrc_handles_by_worker[key]

    def _get_param(
        self,
        col: str,
        table,
        override,
        fallback,
        *,
        override_name: str,
        cli_arg: str,
    ) -> torch.Tensor:
        """Return a per-particle float tensor for a STAR column.

        Args:
            col: STAR column name.
            table: A pandas DataFrame that may contain ``col``.
            override: Optional scalar (or 1D tensor) that overrides the STAR value.
            fallback: Fallback scalar/tensor used when ``col`` is missing and
                ``override`` is not set.
            override_name: Human-readable config key used for error reporting.
            cli_arg: CLI argument used for error reporting.

        Returns:
            A 1D ``float32`` tensor of shape ``(N,)``.

        Raises:
            ValueError: If ``col`` is missing and no override/fallback is provided.
        """
        n = len(table)

        if isinstance(override, torch.Tensor):
            return override

        if override is not None:
            return torch.full((n,), float(override), dtype=torch.float32)

        if col in table:
            out = torch.tensor(table[col].values, dtype=torch.float32)
            if torch.any(~torch.isfinite(out)):
                if isinstance(fallback, torch.Tensor):
                    fb = fallback.to(dtype=torch.float32)
                    if fb.numel() == 1:
                        fb = fb.expand_as(out)
                    out = torch.where(torch.isfinite(out), out, fb)
                elif fallback is not None:
                    out = torch.nan_to_num(
                        out,
                        nan=float(fallback),
                        posinf=float(fallback),
                        neginf=float(fallback),
                    )
            return out

        if isinstance(fallback, torch.Tensor):
            return fallback

        if fallback is None:
            raise ValueError(
                f"Missing STAR column '{col}' and {override_name} is not set. "
                f"Please pass {cli_arg} (or set it in the config file)."
            )

        return torch.full((n,), float(fallback), dtype=torch.float32)

    def _load_ctf_params(self) -> None:
        """Load per-particle CTF parameters.

        This method constructs 1D ``float32`` tensors of shape ``(N,)`` for each
        CTF parameter, using STAR columns when available and falling back to
        user-provided defaults when necessary.
        """
        self.kV = self._get_param(
            "rlnVoltage",
            self.df_particles,
            self.default_optic_params.get("kV"),
            self._optic_fallback_params["kV"],
            override_name="default_optic_params['kV']",
            cli_arg="--kV",
        )
        self.Cs = self._get_param(
            "rlnSphericalAberration",
            self.df_particles,
            self.default_optic_params.get("Cs"),
            self._optic_fallback_params["Cs"],
            override_name="default_optic_params['Cs']",
            cli_arg="--Cs",
        )
        self.Bfac = self._get_param(
            "rlnCtfBfactor",
            self.df_particles,
            self.default_optic_params.get("Bfac"),
            self._optic_fallback_params["Bfac"],
            override_name="default_optic_params['Bfac']",
            cli_arg="--Bfac",
        )
        self.scale = self._get_param(
            "rlnCtfScalefactor",
            self.df_particles,
            self.default_optic_params.get("scale"),
            self._optic_fallback_params["scale"],
            override_name="default_optic_params['scale']",
            cli_arg="--scale",
        )
        self.Q0 = self._get_param(
            "rlnAmplitudeContrast",
            self.df_particles,
            self.default_optic_params.get("Q0"),
            self._optic_fallback_params["Q0"],
            override_name="default_optic_params['Q0']",
            cli_arg="--Q0",
        )
        self.phase_shift = self._get_param(
            "rlnPhaseShift",
            self.df_particles,
            self.default_optic_params.get("phase_shift"),
            self._optic_fallback_params["phase_shift"],
            override_name="default_optic_params['phase_shift']",
            cli_arg="--phase-shift",
        )

        self.DeltafU = self._get_param(
            "rlnDefocusU",
            self.df_particles,
            self.default_particle_params.get("DeltafU"),
            self._particle_fallback_params["DeltafU"],
            override_name="default_particle_params['DeltafU']",
            cli_arg="--DeltafU",
        )
        self.DeltafV = self._get_param(
            "rlnDefocusV",
            self.df_particles,
            self.default_particle_params.get("DeltafV"),
            self.DeltafU,
            override_name="default_particle_params['DeltafV']",
            cli_arg="--DeltafV",
        )
        self.azimuthal_angle = self._get_param(
            "rlnDefocusAngle",
            self.df_particles,
            self.default_particle_params.get("azimuthal_angle"),
            self._particle_fallback_params["azimuthal_angle"],
            override_name="default_particle_params['azimuthal_angle']",
            cli_arg="--defocus-angle",
        )

    def _get_ctf_params_tensor(self, i: int) -> torch.Tensor:
        """Return per-particle CTF parameters as a single tensor.

        Args:
            i (int): Particle index.

        Returns:
            torch.Tensor: A 1D tensor of shape ``(9,)`` with the convention:
            ``[kV, Cs, Bfac, scale, Q0, phase_shift, DeltafU, DeltafV, azimuthal_angle]``.
        """
        return torch.stack(
            [
                self.kV[i],
                self.Cs[i],
                self.Bfac[i],
                self.scale[i],
                self.Q0[i],
                self.phase_shift[i],
                self.DeltafU[i],
                self.DeltafV[i],
                self.azimuthal_angle[i],
            ],
            dim=0,
        )  # (9,)

    @staticmethod
    def _maybe_constant_scalar(x: torch.Tensor) -> float | None:
        """Return the scalar value if a tensor is constant.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            float | None: ``float(x[0])`` if all elements are equal, otherwise
            ``None``.
        """
        if x.numel() == 0:
            return None
        first = x.reshape(-1)[0]
        if bool(torch.allclose(x, first.expand_as(x))):
            return float(first.item())
        return None

    def constant_ctf_params(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return CTF parameters that are constant across the dataset.

        Returns:
            ``(optic, particle)`` where each is a dict mapping parameter name to
            a scalar value. Parameters that vary across particles are omitted.
        """
        optic = {
            "kV": self._maybe_constant_scalar(self.kV),
            "Cs": self._maybe_constant_scalar(self.Cs),
            "Bfac": self._maybe_constant_scalar(self.Bfac),
            "scale": self._maybe_constant_scalar(self.scale),
            "Q0": self._maybe_constant_scalar(self.Q0),
            "phase_shift": self._maybe_constant_scalar(self.phase_shift),
        }
        particle = {
            "DeltafU": self._maybe_constant_scalar(self.DeltafU),
            "DeltafV": self._maybe_constant_scalar(self.DeltafV),
            "azimuthal_angle": self._maybe_constant_scalar(self.azimuthal_angle),
        }
        optic_out = {k: v for k, v in optic.items() if v is not None}
        particle_out = {k: v for k, v in particle.items() if v is not None}
        return optic_out, particle_out

    def populate_data_config(self, data_cfg) -> None:
        """Populate a mutable data config object with dataset-derived values.

        This helper is intended for CLI/config workflows where the dataset is the
        source of truth for values like ``num_particles`` and ``angpix``.

        Args:
            data_cfg: A mutable config object (typically
                :class:`cryoseed.config.DataConfig`).
        """
        if hasattr(data_cfg, "num_particles"):
            data_cfg.num_particles = int(len(self))
        if hasattr(data_cfg, "image_size"):
            data_cfg.image_size = int(self.image_size)
        if hasattr(data_cfg, "angpix"):
            data_cfg.angpix = float(self.effective_angpix())

        if hasattr(data_cfg, "particle_diameter"):
            v = getattr(data_cfg, "particle_diameter")
            if v is None or float(v) <= 0:
                image_size = getattr(data_cfg, "image_size", None)
                if image_size is None or int(image_size) <= 0:
                    image_size = int(self.image_size)

                angpix = getattr(data_cfg, "angpix", None)
                if angpix is None or float(angpix) <= 0:
                    angpix = float(self.effective_angpix())

                data_cfg.particle_diameter = 0.5 * float(image_size) * float(angpix)

        optic_const, particle_const = self.constant_ctf_params()

        if hasattr(data_cfg, "default_optic_params") and isinstance(
            getattr(data_cfg, "default_optic_params"), dict
        ):
            for k, v in optic_const.items():
                if data_cfg.default_optic_params.get(k) is None:
                    data_cfg.default_optic_params[k] = v

        if hasattr(data_cfg, "default_particle_params") and isinstance(
            getattr(data_cfg, "default_particle_params"), dict
        ):
            for k, v in particle_const.items():
                if data_cfg.default_particle_params.get(k) is None:
                    data_cfg.default_particle_params[k] = v

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int | float | None]:
        """Load a single particle image and its metadata.

        Args:
            i (int): Dataset index.

        Returns:
            dict[str, torch.Tensor | int | float | None]: Per-sample dictionary
            consumed by :func:`cryoseed.data.data_collate_fn`.

            - ``image_real``: Real-space image tensor of shape ``(D, D)``.
            - ``particle_index``: The dataset index ``i``.
            - ``stack_index``: Index inside the MRC/MRCS stack.
            - ``euler``: Euler angles in radians (or ``None`` if unavailable).
            - ``trans``: Translation in pixels (or ``None`` if unavailable).
            - ``ctf_params``: CTF parameter vector of shape ``(9,)``.
            - ``angpix``: Effective pixel size in Angstroms/pixel.

        Raises:
            IndexError: If the parsed ``stack_index`` is out of range for the
                corresponding stack.
            RuntimeError: If the MRC/MRCS stack cannot be read.
        """
        path, stack_index = self.samples[i]
        mrcs = self._get_mrc_handle(path)
        data = mrcs.data
        if data is None:
            raise RuntimeError(
                f"MRC/MRCS file '{path}' has no readable data (mrcfile returned data=None). "
                "The file may be truncated/corrupted, not an MRC/MRCS stack, or contains an invalid header."
            )

        if data.ndim > 2:
            if int(stack_index) < 0 or int(stack_index) >= int(data.shape[0]):
                raise IndexError(
                    f"stack_index={int(stack_index)} is out of range for '{path}' with stack size {int(data.shape[0])}"
                )
            img_real_np = np.asarray(data[stack_index], dtype=np.float32)
        else:
            img_real_np = np.asarray(data, dtype=np.float32)

        if not img_real_np.flags.writeable:
            img_real_np = img_real_np.copy()

        image_real = torch.from_numpy(img_real_np)  # (D, D)

        angpix_eff = float(self.effective_angpix())

        if self.image_size < self.full_image_size:
            image = primal_to_fourier_2d(image_real)
            image = downsample2d(image, self.image_size)
            mask = circular_mask(
                int(self.image_size),
                int(self.image_size),
                radius=int(self.image_size) // 2,
                device=image.device,
            )
            image = image.masked_fill(~mask, 0)
            image_real = fourier_to_primal_2d(image).real

        if self.shift_A is None:
            trans = None
        else:
            trans_px = (self.shift_A[i] / angpix_eff).astype(np.float32)
            trans = torch.from_numpy(trans_px)  # (2,)

        if self.euler_rad is None:
            euler = None
        else:
            euler = from_relion_euler(torch.from_numpy(self.euler_rad[i]))  # (3,)

        return {
            "image_real": image_real,
            "particle_index": int(i),
            "stack_index": int(stack_index),
            "euler": euler,
            "trans": trans,
            "ctf_params": self._get_ctf_params_tensor(i),  # (9,)
            "angpix": float(angpix_eff),
        }