from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any


OPTIC_PARAM_KEY_ALIASES: dict[str, str] = {
    "kV": "voltage_kv",
    "Cs": "spherical_aberration_mm",
    "Bfac": "bfactor",
    "scale": "ctf_scale",
    "Q0": "amplitude_contrast",
    "phase_shift": "phase_shift_deg",
}

PARTICLE_PARAM_KEY_ALIASES: dict[str, str] = {
    "DeltafU": "defocus_u_angstrom",
    "DeltafV": "defocus_v_angstrom",
    "azimuthal_angle": "defocus_angle_deg",
}


def _normalize_param_keys(
    params: Any,
    aliases: dict[str, str],
    preferred_keys: list[str],
) -> dict[str, Any]:
    normalized = dict(params or {})
    for old_key, new_key in aliases.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]
        normalized.pop(old_key, None)

    out: dict[str, Any] = {}
    for key in preferred_keys:
        out[key] = normalized.get(key)
    return out


@dataclass
class IOConfig:
    """Input / output paths."""

    star_path: str = ""
    data_path: str = ""
    ref_volume_path: str = ""
    output_path: str = "outputs"
    ssd_cache_root: str = ""


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""

    batch_size: int = 4
    num_workers: int = 0

    num_particles: int = 0
    image_size: int = 0
    angpix: float = 0.0

    particle_diameter: float | None = None

    default_optic_params: dict[str, Any] = field(
        default_factory=lambda: {
            "voltage_kv": None,
            "spherical_aberration_mm": None,
            "bfactor": None,
            "ctf_scale": None,
            "amplitude_contrast": None,
            "phase_shift_deg": None,
        }
    )

    default_particle_params: dict[str, Any] = field(
        default_factory=lambda: {
            "defocus_u_angstrom": None,
            "defocus_v_angstrom": None,
            "defocus_angle_deg": None,
        }
    )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = ""
    log_prefix: str = "cryoseed"


@dataclass
class ReconstructionConfig:
    """Reconstruction / volume update behavior."""

    num_volumes: int = 1

    external_reconstruct: bool = False
    full_backprojection: bool = False
    requires_grad: bool = False
    requires_accum: bool = True

    backproject_chunk: int = 16384
    accumulate_chunk: int = 16384


@dataclass
class StatisticsConfig:
    """Noise / prior statistics."""

    use_noise: bool = True
    use_prior: bool = False

    init_variance: float = 1.0
    precision_eps: float = 1e-6
    tail_floor: float = 1e-5
    init_lowpass_cutoff: int | None = None


@dataclass
class RefinementConfig:
    """Outer refinement loop configuration."""

    num_epochs: int = 50
    fsc_threshold: float = 0.143
    init_lowpass_angstrom: float = 30

@dataclass
class SchedulerConfig:
    """Scheduling / resolution control."""

    # confidence-driven update
    confidence_threshold: float = 0.1
    fsc_resolution_patience: int = 3
    fsc_resolution_improvement_threshold: float = 0.0
    fsc_resolution_rebound_threshold: float = 1e-2

    # side_length update policy
    increase_radius_step: int = 10
    increase_radius_aggressive_factor: float = 0.25

    # Starting HEALPix order used while the scheduler stays in global search
    base_healpix_order: int = 3
    # HEALPix order at which the scheduler switches from global to local search
    auto_local_healpix_order: int = 4

    # cache configuration
    use_cache: bool = False
    cache_max_healpix_order: int = 4
    ssd_cache_min_side_length: int = 150
    trans_extent_scale: float = 3.0

@dataclass
class PoseSearchConfig:
    """Pose search configuration."""

    init_healpix_order: int = 2

    neighbor_steps: int = 2
    init_trans_grid_extent: float = 5
    trans_grid_samples: int = 5
    trans_grid_x_shift: int = 0
    trans_grid_y_shift: int = 0

    pose_chunk_factor: int | None = 2560
    max_candidates: int = 100
    mse_chunk: int = 8192
    candidate_select_threshold: float = 0.999
    renormalize_sel_prob: bool = True
    oversampling_deduplicate: bool = False


@dataclass
class ReproduceConfig:
    """Reproducibility configuration."""

    seed: int = 42
    deterministic: bool = False


@dataclass
class MainConfig:
    """Top-level application config."""

    io: IOConfig = field(default_factory=IOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    pose_search: PoseSearchConfig = field(default_factory=PoseSearchConfig)
    reproduce: ReproduceConfig = field(default_factory=ReproduceConfig)

    def __post_init__(self) -> None:
        if not self.io.ssd_cache_root:
            self.io.ssd_cache_root = str(Path(self.io.output_path) / "ssd_cache")
        if not self.logging.log_dir:
            self.logging.log_dir = str(Path(self.io.output_path) / "logs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _yaml_scalar_repr(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            if value == "":
                return '""'
            if value.strip() != value or any(ch in value for ch in ':\'"#{}[]&,*!?|<>=%@`\\\n\r\t'):
                return json.dumps(value, ensure_ascii=False)
            return value
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _to_yaml_lines(cls, value: Any, *, indent: int = 0) -> list[str]:
        prefix = " " * indent

        if isinstance(value, dict):
            if not value:
                return [f"{prefix}{{}}"]

            lines: list[str] = []
            for key, item in value.items():
                rendered_key = str(key)
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{rendered_key}:")
                    lines.extend(cls._to_yaml_lines(item, indent=indent + 2))
                else:
                    lines.append(f"{prefix}{rendered_key}: {cls._yaml_scalar_repr(item)}")
            return lines

        if isinstance(value, list):
            if not value:
                return [f"{prefix}[]"]

            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    nested = cls._to_yaml_lines(item, indent=indent + 2)
                    lines.append(f"{prefix}- {nested[0].lstrip()}")
                    lines.extend(nested[1:])
                else:
                    lines.append(f"{prefix}- {cls._yaml_scalar_repr(item)}")
            return lines

        return [f"{prefix}{cls._yaml_scalar_repr(value)}"]

    @classmethod
    def _to_yaml_text(cls, value: Any) -> str:
        return "\n".join(cls._to_yaml_lines(value)) + "\n"

    @staticmethod
    def _make_timestamped_path(path: Path, timestamp: str) -> Path:
        suffix = "".join(path.suffixes)
        stem = path.name[: -len(suffix)] if suffix else path.name

        candidate = path.with_name(f"{stem}.{timestamp}{suffix}")
        index = 1
        while candidate.exists():
            candidate = path.with_name(f"{stem}.{timestamp}.{index}{suffix}")
            index += 1
        return candidate

    def save_output_config(self, filename: str = "config.yml") -> tuple[Path, Path]:
        output_dir = Path(self.io.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        target_path = output_dir / filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        snapshot_path = self._make_timestamped_path(target_path, timestamp)
        yaml_text = self._to_yaml_text(self.to_dict())

        target_path.write_text(yaml_text, encoding="utf-8")
        snapshot_path.write_text(yaml_text, encoding="utf-8")
        return target_path, snapshot_path


    @staticmethod
    def _filter_kwargs(cls, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in data.items() if k in allowed}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: str | None = None) -> "MainConfig":
        if not isinstance(data, dict):
            raise TypeError(f"config root must be a dict, got {type(data)}")

        normalized = dict(data)

        if isinstance(normalized.get("io"), dict):
            normalized["io"] = IOConfig(**cls._filter_kwargs(IOConfig, normalized["io"]))
        else:
            normalized["io"] = IOConfig()

        if isinstance(normalized.get("data"), dict):
            data_config = dict(normalized["data"])
            data_config["default_optic_params"] = _normalize_param_keys(
                data_config.get("default_optic_params"),
                OPTIC_PARAM_KEY_ALIASES,
                [
                    "voltage_kv",
                    "spherical_aberration_mm",
                    "bfactor",
                    "ctf_scale",
                    "amplitude_contrast",
                    "phase_shift_deg",
                ],
            )
            data_config["default_particle_params"] = _normalize_param_keys(
                data_config.get("default_particle_params"),
                PARTICLE_PARAM_KEY_ALIASES,
                [
                    "defocus_u_angstrom",
                    "defocus_v_angstrom",
                    "defocus_angle_deg",
                ],
            )
            normalized["data"] = DataConfig(**cls._filter_kwargs(DataConfig, data_config))
        else:
            normalized["data"] = DataConfig()

        if isinstance(normalized.get("logging"), dict):
            normalized["logging"] = LoggingConfig(**cls._filter_kwargs(LoggingConfig, normalized["logging"]))
        else:
            normalized["logging"] = LoggingConfig()

        if isinstance(normalized.get("reconstruction"), dict):
            normalized["reconstruction"] = ReconstructionConfig(
                **cls._filter_kwargs(ReconstructionConfig, normalized["reconstruction"])
            )
        else:
            normalized["reconstruction"] = ReconstructionConfig()

        if isinstance(normalized.get("statistics"), dict):
            normalized["statistics"] = StatisticsConfig(
                **cls._filter_kwargs(StatisticsConfig, normalized["statistics"])
            )
        else:
            normalized["statistics"] = StatisticsConfig()

        if isinstance(normalized.get("refinement"), dict):
            normalized["refinement"] = RefinementConfig(
                **cls._filter_kwargs(RefinementConfig, normalized["refinement"])
            )
        else:
            normalized["refinement"] = RefinementConfig()

        if isinstance(normalized.get("scheduler"), dict):
            scheduler_data = dict(normalized["scheduler"])
            if (
                "auto_local_order" in scheduler_data
                and "auto_local_healpix_order" not in scheduler_data
            ):
                scheduler_data["auto_local_healpix_order"] = scheduler_data["auto_local_order"]
            normalized["scheduler"] = SchedulerConfig(
                **cls._filter_kwargs(SchedulerConfig, scheduler_data)
            )
        else:
            normalized["scheduler"] = SchedulerConfig()

        if isinstance(normalized.get("pose_search"), dict):
            normalized["pose_search"] = PoseSearchConfig(
                **cls._filter_kwargs(PoseSearchConfig, normalized["pose_search"])
            )
        else:
            normalized["pose_search"] = PoseSearchConfig()

        if isinstance(normalized.get("reproduce"), dict):
            normalized["reproduce"] = ReproduceConfig(
                **cls._filter_kwargs(ReproduceConfig, normalized["reproduce"])
            )
        else:
            normalized["reproduce"] = ReproduceConfig()

        cfg = cls(**cls._filter_kwargs(cls, normalized))

        if base_dir is not None:
            base = Path(base_dir)

            if cfg.io.star_path and not Path(cfg.io.star_path).is_absolute():
                cfg.io.star_path = str((base / cfg.io.star_path).resolve())

            if cfg.io.output_path and not Path(cfg.io.output_path).is_absolute():
                cfg.io.output_path = str((base / cfg.io.output_path).resolve())

            if cfg.logging.log_dir and not Path(cfg.logging.log_dir).is_absolute():
                cfg.logging.log_dir = str((base / cfg.logging.log_dir).resolve())

            if cfg.io.data_path and not Path(cfg.io.data_path).is_absolute():
                cfg.io.data_path = str((base / cfg.io.data_path).resolve())

            if cfg.io.ref_volume_path and not Path(cfg.io.ref_volume_path).is_absolute():
                cfg.io.ref_volume_path = str((base / cfg.io.ref_volume_path).resolve())

            if cfg.io.ssd_cache_root and not Path(cfg.io.ssd_cache_root).is_absolute():
                cfg.io.ssd_cache_root = str((base / cfg.io.ssd_cache_root).resolve())


        return cfg

    @staticmethod
    def _load_file(path: str) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        text = p.read_text()
        suffix = p.suffix.lower()

        if suffix == ".json":
            out = json.loads(text)
            return out if isinstance(out, dict) else {}

        yaml_data = None
        try:
            from ruamel.yaml import YAML

            y = YAML(typ="safe")
            yaml_data = y.load(text)
        except Exception:
            yaml_data = None

        if yaml_data is None:
            try:
                import yaml as _pyyaml

                yaml_data = _pyyaml.safe_load(text)
            except Exception:
                yaml_data = None

        if isinstance(yaml_data, dict):
            return yaml_data

        out = json.loads(text)
        return out if isinstance(out, dict) else {}

    @classmethod
    def from_file(cls, path: str) -> "MainConfig":
        data = cls._load_file(path)
        return cls.from_dict(data, base_dir=str(Path(path).resolve().parent))

    @classmethod
    def from_cli_args(cls, args) -> "MainConfig":
        config_path = getattr(args, "config", None)
        cfg = cls.from_file(config_path) if config_path else cls()

        base_dir = str(Path(config_path).resolve().parent) if config_path else None

        initial_output_path = cfg.io.output_path
        # Determine which field names are duplicated across sections.
        occurrences: dict[str, list[str]] = {}
        for section in fields(cls):
            section_obj = getattr(cfg, section.name)
            for f in fields(section_obj.__class__):
                occurrences.setdefault(f.name, []).append(section.name)

        duplicated_fields = {name for name, secs in occurrences.items() if len(secs) > 1}

        # Flat override policy:
        # - If a field name is unique, read from --field
        # - If duplicated, read from --section_field
        # Also supports dict field override via JSON string.
        for section in fields(cls):
            section_obj = getattr(cfg, section.name)
            for f in fields(section_obj.__class__):
                arg_name = f"{section.name}_{f.name}" if f.name in duplicated_fields else f.name
                if not hasattr(args, arg_name):
                    continue

                v = getattr(args, arg_name)
                if v is None:
                    continue

                current = getattr(section_obj, f.name)
                if isinstance(current, dict):
                    if isinstance(v, str):
                        parsed = json.loads(v)
                        if parsed is None:
                            parsed = {}
                        if not isinstance(parsed, dict):
                            raise TypeError(f"{arg_name} must be a JSON object")
                        v = parsed
                    elif not isinstance(v, dict):
                        raise TypeError(f"{arg_name} must be a dict or a JSON object string")

                setattr(section_obj, f.name, v)

        # Single-key overrides for nested defaults
        optic_keys = [
            "voltage_kv",
            "spherical_aberration_mm",
            "bfactor",
            "ctf_scale",
            "amplitude_contrast",
            "phase_shift_deg",
        ]
        for key in optic_keys:
            arg_name = f"default_optic_params_{key}"
            if hasattr(args, arg_name):
                v = getattr(args, arg_name)
                if v is not None:
                    cfg.data.default_optic_params[key] = float(v)

        particle_keys = ["defocus_u_angstrom", "defocus_v_angstrom"]
        for key in particle_keys:
            arg_name = f"default_particle_params_{key}"
            if hasattr(args, arg_name):
                v = getattr(args, arg_name)
                if v is not None:
                    cfg.data.default_particle_params[key] = float(v)

        az_arg = "default_particle_params_defocus_angle_deg"
        if hasattr(args, az_arg):
            v = getattr(args, az_arg)
            if v is not None:
                cfg.data.default_particle_params["defocus_angle_deg"] = float(v)

        cfg.data.default_optic_params = _normalize_param_keys(
            cfg.data.default_optic_params,
            OPTIC_PARAM_KEY_ALIASES,
            [
                "voltage_kv",
                "spherical_aberration_mm",
                "bfactor",
                "ctf_scale",
                "amplitude_contrast",
                "phase_shift_deg",
            ],
        )
        cfg.data.default_particle_params = _normalize_param_keys(
            cfg.data.default_particle_params,
            PARTICLE_PARAM_KEY_ALIASES,
            [
                "defocus_u_angstrom",
                "defocus_v_angstrom",
                "defocus_angle_deg",
            ],
        )

        output_path_arg = getattr(args, "output_path", None)
        output_path_changed = output_path_arg is not None and cfg.io.output_path != initial_output_path

        log_dir_arg = getattr(args, "log_dir", None)
        if log_dir_arg is None and (
            (not cfg.logging.log_dir)
            or (output_path_changed and cfg.logging.log_dir == str(Path(initial_output_path) / "logs"))
        ):
            cfg.logging.log_dir = str(Path(cfg.io.output_path) / "logs")

        ssd_cache_root_arg = getattr(args, "ssd_cache_root", None)
        if ssd_cache_root_arg is None and (
            (not cfg.io.ssd_cache_root)
            or (output_path_changed and cfg.io.ssd_cache_root == str(Path(initial_output_path) / "ssd_cache"))
        ):
            cfg.io.ssd_cache_root = str(Path(cfg.io.output_path) / "ssd_cache")

        if base_dir is not None:
            base = Path(base_dir)

            if cfg.io.star_path and not Path(cfg.io.star_path).is_absolute():
                cfg.io.star_path = str((base / cfg.io.star_path).resolve())

            if cfg.io.output_path and not Path(cfg.io.output_path).is_absolute():
                cfg.io.output_path = str((base / cfg.io.output_path).resolve())

            if cfg.logging.log_dir and not Path(cfg.logging.log_dir).is_absolute():
                cfg.logging.log_dir = str((base / cfg.logging.log_dir).resolve())

            if cfg.io.data_path and not Path(cfg.io.data_path).is_absolute():
                cfg.io.data_path = str((base / cfg.io.data_path).resolve())

            if cfg.io.ref_volume_path and not Path(cfg.io.ref_volume_path).is_absolute():
                cfg.io.ref_volume_path = str((base / cfg.io.ref_volume_path).resolve())

            if cfg.io.ssd_cache_root and not Path(cfg.io.ssd_cache_root).is_absolute():
                cfg.io.ssd_cache_root = str((base / cfg.io.ssd_cache_root).resolve())

        cfg.__post_init__()
        return cfg