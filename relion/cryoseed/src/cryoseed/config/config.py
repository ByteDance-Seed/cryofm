from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from cryoseed.utils import particle_mask as particle_mask_utils


def _normalize_param_keys(
    params: Any,
    preferred_keys: list[str],
    *,
    location: str | None = None,
) -> dict[str, Any]:
    if params is None:
        normalized: dict[str, Any] = {}
    elif isinstance(params, dict):
        normalized = dict(params)
    else:
        detail = f" `{location}`" if location else ""
        raise TypeError(f"config mapping{detail} must be a dict, got {type(params)}")

    unknown_keys = sorted(k for k in normalized if k not in preferred_keys)
    if unknown_keys:
        unknown_key = unknown_keys[0]
        detail = f"{location}.{unknown_key}" if location else unknown_key
        suggestion = difflib.get_close_matches(unknown_key, preferred_keys, n=1)
        msg = f"Unknown config key `{detail}`."
        if suggestion:
            target = f"{location}.{suggestion[0]}" if location else suggestion[0]
            msg += f" Did you mean `{target}`?"
        raise ValueError(msg)

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
class ParticleMaskConfig:
    """Real-space particle masking used for pose search."""

    enabled: bool = False
    zero_mask: bool = True
    soft_edge_pixels: float = 5.0
    protection_disable_epochs: int = 5
    protection_radius_factor: float = (
        particle_mask_utils.DEFAULT_PARTICLE_MASK_PROTECTION_RADIUS_FACTOR
    )


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""

    batch_size: int = 4
    num_workers: int = 0

    num_particles: int = 0
    image_size: int = 0
    angpix: float = 0.0

    particle_diameter: float | None = None
    particle_mask: ParticleMaskConfig = field(default_factory=ParticleMaskConfig)

    default_optic_params: dict[str, Any] = field(
        default_factory=lambda: {
            "voltage_kv": None,
            "spherical_aberration_mm": None,
            "ctf_bfactor": None,
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
    level: str = "INFO"


@dataclass
class ReconstructionConfig:
    """Reconstruction / volume update behavior."""

    num_volumes: int = 1

    external_reconstruct: bool = False
    full_backprojection: bool = False

    backproject_chunk: int = 16384
    accumulate_chunk: int = 16384


@dataclass
class StatisticsConfig:
    """Noise / prior statistics."""

    use_noise: bool = True
    use_prior: bool = True

    init_variance: float = 1.0
    precision_eps: float = 1e-6
    noise_ema_decay: float = 0.0
    noise_prior_weight: float = 0.0
    noise_inflated_weight: float = 0.0
    noise_inflated_decay: float | None = None
    noise_inflated_scale: float = 8.0
    tail_floor: float = 1e-5
    init_lowpass_cutoff: int | None = None


@dataclass
class HomoRefineConfig:
    """Homogeneous refinement configuration."""

    num_epochs: int = 50
    # Enable an NCC-based correlation criterion on the first epoch only.
    first_epoch_ncc: bool = True
    fsc_threshold: float = 0.143
    init_lowpass_angstrom: float = 30
    solvent_mask: str = "none"
    solvent_mask_soft_edge_pixels: float = 5.0
    solvent_fsc_correction: bool = False

@dataclass
class SchedulerConfig:
    """Scheduling / resolution control."""

    # confidence-driven update
    schedule_check_interval_iters: int = 50
    confidence_threshold: float = 0.1
    convergence_patience: int = 3
    fsc_resolution_improvement_threshold: float = 5e-3
    fsc_resolution_rebound_threshold: float = 1e-2
    trans_update_rms_threshold: float = 0.5
    pose_translation_center_mode: str = "auto"

    # side_length update policy
    increase_radius_step: int = 10
    increase_radius_aggressive_factor: float = 0.25
    increase_radius_aggressive_fsc_threshold: float = 0.2

    # Starting HEALPix order used while the scheduler stays in global search
    base_healpix_order: int = 3
    # HEALPix order at which the scheduler switches from global to local search
    auto_local_healpix_order: int = 4
    # Block the auto-local switch until assignment change is small enough.
    auto_local_assignment_change_threshold: float = 0.05

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
    init_trans_grid_extent: float | None = None
    trans_grid_samples: int = 5
    trans_grid_x_shift: int = 0
    trans_grid_y_shift: int = 0

    pose_chunk_factor: int | None = 2560
    max_candidates: int = -1
    criterion_chunk: int = 8192
    candidate_select_threshold: float = 0.999
    volume_class_similarity: float = 0.0
    volume_class_similarity_scope: str = "global"
    oversampling_deduplicate: bool = False
    ring_averaged_mse: bool = False


@dataclass
class ReproduceConfig:
    """Reproducibility configuration."""

    seed: int = 42
    deterministic: bool = False


@dataclass
class AbInitioConfig:
    """Ab initio reconstruction configuration."""

    num_epochs: int = 1000
    init_particles_per_volume: int = 100
    init_lowpass_angstrom: float = 50
    solvent_mask: str = "none"
    solvent_mask_soft_edge_pixels: float = 5.0
    learning_rate: float = 1.0
    learning_rate_decay: float = 0.9995
    momentum: float = 0.9
    loss_ema_decay: float = 0.9
    pose_rms_ema_decay: float = 0.9
    target_side_length_resolution: float = 10.0
    target_healpix_order: int | None = None
    pose_rotation_stability_factor: float = 1.0
    pose_translation_stability_factor: float = 0.5


FieldLocation = tuple[str, str]


ABINITIO_HIDDEN_FIELDS: set[FieldLocation] = {
    ("reconstruction", "external_reconstruct"),
    ("homorefine", "num_epochs"),
    ("homorefine", "first_epoch_ncc"),
    ("homorefine", "fsc_threshold"),
    ("homorefine", "init_lowpass_angstrom"),
    ("homorefine", "solvent_mask"),
    ("homorefine", "solvent_mask_soft_edge_pixels"),
    ("homorefine", "solvent_fsc_correction"),
    ("scheduler", "fsc_resolution_improvement_threshold"),
    ("scheduler", "fsc_resolution_rebound_threshold"),
    ("scheduler", "increase_radius_aggressive_factor"),
    ("scheduler", "increase_radius_aggressive_fsc_threshold"),
    ("scheduler", "base_healpix_order"),
    ("scheduler", "auto_local_healpix_order"),
    ("scheduler", "use_cache"),
    ("scheduler", "cache_max_healpix_order"),
    ("scheduler", "ssd_cache_min_side_length"),
}


ABINITIO_FIXED_FIELD_VALUES: dict[FieldLocation, object] = {
    ("reconstruction", "external_reconstruct"): False,
    ("scheduler", "fsc_resolution_improvement_threshold"): 5e-3,
    ("scheduler", "fsc_resolution_rebound_threshold"): 1e-2,
    ("scheduler", "increase_radius_aggressive_factor"): 0.25,
    ("scheduler", "increase_radius_aggressive_fsc_threshold"): 0.2,
    ("scheduler", "base_healpix_order"): 3,
    ("scheduler", "auto_local_healpix_order"): 4,
    ("scheduler", "use_cache"): False,
    ("scheduler", "cache_max_healpix_order"): 4,
    ("scheduler", "ssd_cache_min_side_length"): 150,
}


HOMOREFINE_HIDDEN_FIELDS: set[FieldLocation] = {
    ("abinitio", "num_epochs"),
    ("abinitio", "init_particles_per_volume"),
    ("abinitio", "learning_rate"),
    ("abinitio", "momentum"),
    ("abinitio", "loss_ema_decay"),
    ("abinitio", "pose_rms_ema_decay"),
    ("abinitio", "target_side_length_resolution"),
    ("abinitio", "target_healpix_order"),
    ("abinitio", "pose_rotation_stability_factor"),
    ("abinitio", "pose_translation_stability_factor"),
    ("scheduler", "auto_local_assignment_change_threshold"),
}


HOMOREFINE_FIXED_FIELD_VALUES: dict[FieldLocation, object] = {}


COMMAND_HIDDEN_FIELDS: dict[str, set[FieldLocation]] = {
    "abinitio": ABINITIO_HIDDEN_FIELDS,
    "homorefine": HOMOREFINE_HIDDEN_FIELDS,
}


COMMAND_FIXED_FIELD_VALUES: dict[str, dict[FieldLocation, object]] = {
    "abinitio": ABINITIO_FIXED_FIELD_VALUES,
    "homorefine": HOMOREFINE_FIXED_FIELD_VALUES,
}


@dataclass
class MainConfig:
    """Top-level application config."""

    io: IOConfig = field(default_factory=IOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    homorefine: HomoRefineConfig = field(default_factory=HomoRefineConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    pose_search: PoseSearchConfig = field(default_factory=PoseSearchConfig)
    reproduce: ReproduceConfig = field(default_factory=ReproduceConfig)
    abinitio: AbInitioConfig = field(default_factory=AbInitioConfig)

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

    def to_export_dict(self, command: str | None = None) -> dict[str, Any]:
        data = asdict(self)
        if not command:
            return data

        hidden_fields = COMMAND_HIDDEN_FIELDS.get(command, set())
        if not hidden_fields:
            return data

        exported: dict[str, Any] = {}
        for section_name, section_data in data.items():
            if not isinstance(section_data, dict):
                exported[section_name] = section_data
                continue

            hidden_names = {
                field_name
                for hidden_section, field_name in hidden_fields
                if hidden_section == section_name
            }
            visible_section = {
                key: value for key, value in section_data.items() if key not in hidden_names
            }
            if visible_section:
                exported[section_name] = visible_section

        return exported

    def save_output_config(
        self,
        *,
        command: str | None = None,
        filename: str = "config.yml",
    ) -> tuple[Path, Path]:
        output_dir = Path(self.io.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        target_path = output_dir / filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        snapshot_path = self._make_timestamped_path(target_path, timestamp)
        yaml_text = self._to_yaml_text(self.to_export_dict(command))

        target_path.write_text(yaml_text, encoding="utf-8")
        snapshot_path.write_text(yaml_text, encoding="utf-8")
        return target_path, snapshot_path


    @staticmethod
    def _filter_kwargs(cls, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in data.items() if k in allowed}

    @staticmethod
    def _section_types() -> dict[str, type]:
        return {
            "io": IOConfig,
            "data": DataConfig,
            "logging": LoggingConfig,
            "reconstruction": ReconstructionConfig,
            "statistics": StatisticsConfig,
            "homorefine": HomoRefineConfig,
            "scheduler": SchedulerConfig,
            "pose_search": PoseSearchConfig,
            "reproduce": ReproduceConfig,
            "abinitio": AbInitioConfig,
        }

    @classmethod
    def _all_field_locations(cls) -> dict[str, list[str]]:
        locations: dict[str, list[str]] = {}
        for section_name, section_type in cls._section_types().items():
            for field_info in fields(section_type):
                locations.setdefault(field_info.name, []).append(section_name)
        return locations

    @staticmethod
    def _format_suggestions(suggestions: list[str]) -> str:
        if len(suggestions) == 1:
            return f" Did you mean `{suggestions[0]}`?"
        joined = ", ".join(f"`{item}`" for item in suggestions)
        return f" Did you mean one of: {joined}?"

    @classmethod
    def _raise_unknown_section_error(cls, section_name: str) -> None:
        valid_sections = list(cls._section_types())
        suggestion = difflib.get_close_matches(section_name, valid_sections, n=1)
        msg = f"Unknown config section `{section_name}`."
        if suggestion:
            msg += f" Did you mean `{suggestion[0]}`?"
        raise ValueError(msg)

    @classmethod
    def _raise_unknown_field_error(cls, section_name: str, key: str) -> None:
        section_type = cls._section_types()[section_name]
        same_section_fields = [field_info.name for field_info in fields(section_type)]
        other_sections = [
            f"{other_section}.{key}"
            for other_section in cls._all_field_locations().get(key, [])
            if other_section != section_name
        ]
        suggestions = other_sections[:]
        if not suggestions:
            close_match = difflib.get_close_matches(key, same_section_fields, n=1)
            if close_match:
                suggestions = [f"{section_name}.{close_match[0]}"]
            else:
                current_key = f"{section_name}.{key}"
                all_keys = [
                    f"{name}.{field_info.name}"
                    for name, section_cls in cls._section_types().items()
                    for field_info in fields(section_cls)
                ]
                suggestions = difflib.get_close_matches(current_key, all_keys, n=3)

        msg = f"Unknown config key `{section_name}.{key}`."
        if suggestions:
            msg += cls._format_suggestions(suggestions)
        raise ValueError(msg)

    @classmethod
    def _validate_top_level_keys(cls, data: dict[str, Any]) -> None:
        allowed = set(cls._section_types())
        for key in data:
            if key not in allowed:
                cls._raise_unknown_section_error(str(key))

    @classmethod
    def _normalize_section_data(cls, section_name: str, data: Any) -> dict[str, Any]:
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise TypeError(f"config section `{section_name}` must be a dict, got {type(data)}")

        normalized = dict(data)
        allowed = {field_info.name for field_info in fields(cls._section_types()[section_name])}
        for key in normalized:
            if key not in allowed:
                cls._raise_unknown_field_error(section_name, str(key))
        return normalized

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: str | None = None) -> "MainConfig":
        if not isinstance(data, dict):
            raise TypeError(f"config root must be a dict, got {type(data)}")

        normalized = dict(data)
        cls._validate_top_level_keys(normalized)

        if "io" in normalized:
            normalized["io"] = IOConfig(**cls._filter_kwargs(IOConfig, cls._normalize_section_data("io", normalized["io"])))
        else:
            normalized["io"] = IOConfig()

        if "data" in normalized:
            data_config = cls._normalize_section_data("data", normalized["data"])
            particle_mask_data = data_config.get("particle_mask")
            if isinstance(particle_mask_data, dict):
                data_config["particle_mask"] = ParticleMaskConfig(
                    **cls._filter_kwargs(ParticleMaskConfig, particle_mask_data)
                )
            else:
                data_config["particle_mask"] = ParticleMaskConfig()
            data_config["default_optic_params"] = _normalize_param_keys(
                data_config.get("default_optic_params"),
                [
                    "voltage_kv",
                    "spherical_aberration_mm",
                    "ctf_bfactor",
                    "ctf_scale",
                    "amplitude_contrast",
                    "phase_shift_deg",
                ],
                location="data.default_optic_params",
            )
            data_config["default_particle_params"] = _normalize_param_keys(
                data_config.get("default_particle_params"),
                [
                    "defocus_u_angstrom",
                    "defocus_v_angstrom",
                    "defocus_angle_deg",
                ],
                location="data.default_particle_params",
            )
            normalized["data"] = DataConfig(**cls._filter_kwargs(DataConfig, data_config))
        else:
            normalized["data"] = DataConfig()

        if "logging" in normalized:
            logging_data = cls._normalize_section_data("logging", normalized["logging"])
            normalized["logging"] = LoggingConfig(**cls._filter_kwargs(LoggingConfig, logging_data))
        else:
            normalized["logging"] = LoggingConfig()

        if "reconstruction" in normalized:
            reconstruction_data = cls._normalize_section_data("reconstruction", normalized["reconstruction"])
            normalized["reconstruction"] = ReconstructionConfig(
                **cls._filter_kwargs(ReconstructionConfig, reconstruction_data)
            )
        else:
            normalized["reconstruction"] = ReconstructionConfig()

        if "statistics" in normalized:
            statistics_data = cls._normalize_section_data("statistics", normalized["statistics"])
            normalized["statistics"] = StatisticsConfig(
                **cls._filter_kwargs(StatisticsConfig, statistics_data)
            )
        else:
            normalized["statistics"] = StatisticsConfig()

        if "homorefine" in normalized:
            homorefine_data = cls._normalize_section_data("homorefine", normalized["homorefine"])
            normalized["homorefine"] = HomoRefineConfig(
                **cls._filter_kwargs(HomoRefineConfig, homorefine_data)
            )
        else:
            normalized["homorefine"] = HomoRefineConfig()

        if "scheduler" in normalized:
            scheduler_data = cls._normalize_section_data("scheduler", normalized["scheduler"])
            normalized["scheduler"] = SchedulerConfig(
                **cls._filter_kwargs(SchedulerConfig, scheduler_data)
            )
        else:
            normalized["scheduler"] = SchedulerConfig()

        if "pose_search" in normalized:
            pose_search_data = cls._normalize_section_data("pose_search", normalized["pose_search"])
            normalized["pose_search"] = PoseSearchConfig(
                **cls._filter_kwargs(PoseSearchConfig, pose_search_data)
            )
        else:
            normalized["pose_search"] = PoseSearchConfig()

        if "reproduce" in normalized:
            reproduce_data = cls._normalize_section_data("reproduce", normalized["reproduce"])
            normalized["reproduce"] = ReproduceConfig(
                **cls._filter_kwargs(ReproduceConfig, reproduce_data)
            )
        else:
            normalized["reproduce"] = ReproduceConfig()

        if "abinitio" in normalized:
            abinitio_data = cls._normalize_section_data("abinitio", normalized["abinitio"])
            normalized["abinitio"] = AbInitioConfig(
                **cls._filter_kwargs(AbInitioConfig, abinitio_data)
            )
        else:
            normalized["abinitio"] = AbInitioConfig()

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

            cfg.homorefine.solvent_mask = cls._resolve_solvent_mask_path(
                cfg.homorefine.solvent_mask,
                base,
            )
            cfg.abinitio.solvent_mask = cls._resolve_solvent_mask_path(
                cfg.abinitio.solvent_mask,
                base,
            )


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
    def _load_command_defaults(cls, command: str | None) -> dict[str, Any]:
        if not command:
            return {}
        path = Path(__file__).resolve().parent / "defaults" / f"{command}.yaml"
        if not path.is_file():
            return {}
        return cls._load_file(str(path))

    @staticmethod
    def _format_fixed_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return repr(value)

    @classmethod
    def _validate_command_fixed_fields(
        cls,
        command: str | None,
        cfg: "MainConfig",
    ) -> None:
        if not command:
            return
        fixed_fields = COMMAND_FIXED_FIELD_VALUES.get(command, {})
        for (section_name, field_name), expected in fixed_fields.items():
            section_obj = getattr(cfg, section_name)
            actual = getattr(section_obj, field_name)
            if actual != expected:
                location = f"{section_name}.{field_name}"
                expected_repr = cls._format_fixed_value(expected)
                raise ValueError(
                    f"{command} does not support {location}; keep it set to {expected_repr}."
                )

    def validate_for_command(self, command: str | None) -> "MainConfig":
        self._validate_command_fixed_fields(command, self)
        return self

    @staticmethod
    def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = MainConfig._deep_merge_dict(current, value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def from_file(cls, path: str, *, command: str | None = None) -> "MainConfig":
        data = cls._load_file(path)
        cfg = cls.from_dict(data, base_dir=str(Path(path).resolve().parent))
        return cfg.validate_for_command(command)

    @staticmethod
    def _resolve_solvent_mask_path(value: str, base: Path) -> str:
        if value is None:
            return "none"
        value = str(value).strip()
        if not value:
            return "none"
        candidate = Path(value)
        if candidate.is_absolute():
            return str(candidate)
        reserved = {"none", "sphere", "auto"}
        if value.lower() in reserved:
            return value.lower()
        return str((base / candidate).resolve())

    @classmethod
    def from_cli_args(cls, args) -> "MainConfig":
        command = getattr(args, "command", None)
        config_path = getattr(args, "config", None)
        merged_data = cls._load_command_defaults(command)
        base_dir = None
        if config_path:
            user_data = cls._load_file(config_path)
            merged_data = cls._deep_merge_dict(merged_data, user_data)
            base_dir = str(Path(config_path).resolve().parent)
        cfg = cls.from_dict(merged_data, base_dir=base_dir)

        initial_output_path = cfg.io.output_path
        # Determine which field names are duplicated across sections.
        hidden_fields = COMMAND_HIDDEN_FIELDS.get(command, set())
        occurrences: dict[str, list[str]] = {}
        for section in fields(cls):
            section_obj = getattr(cfg, section.name)
            for f in fields(section_obj.__class__):
                if (section.name, f.name) in hidden_fields:
                    continue
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
            "ctf_bfactor",
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

        if hasattr(args, "particle_mask_enabled"):
            cfg.data.particle_mask.enabled = bool(getattr(args, "particle_mask_enabled"))
        if hasattr(args, "particle_mask_zero_mask"):
            cfg.data.particle_mask.zero_mask = bool(getattr(args, "particle_mask_zero_mask"))
        if hasattr(args, "particle_mask_soft_edge_pixels"):
            v = getattr(args, "particle_mask_soft_edge_pixels")
            if v is not None:
                cfg.data.particle_mask.soft_edge_pixels = float(v)
        if hasattr(args, "particle_mask_protection_disable_epochs"):
            v = getattr(args, "particle_mask_protection_disable_epochs")
            if v is not None:
                cfg.data.particle_mask.protection_disable_epochs = int(v)
        if hasattr(args, "particle_mask_protection_coverage"):
            v = getattr(args, "particle_mask_protection_coverage")
            if v is not None:
                cfg.data.particle_mask.protection_radius_factor = (
                    particle_mask_utils.PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS[str(v)]
                )
        if hasattr(args, "particle_mask_protection_radius_factor"):
            v = getattr(args, "particle_mask_protection_radius_factor")
            if v is not None:
                cfg.data.particle_mask.protection_radius_factor = float(v)

        cfg.data.default_optic_params = _normalize_param_keys(
            cfg.data.default_optic_params,
            [
                "voltage_kv",
                "spherical_aberration_mm",
                "ctf_bfactor",
                "ctf_scale",
                "amplitude_contrast",
                "phase_shift_deg",
            ],
        )
        cfg.data.default_particle_params = _normalize_param_keys(
            cfg.data.default_particle_params,
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

            cfg.homorefine.solvent_mask = cls._resolve_solvent_mask_path(
                cfg.homorefine.solvent_mask,
                base,
            )
            cfg.abinitio.solvent_mask = cls._resolve_solvent_mask_path(
                cfg.abinitio.solvent_mask,
                base,
            )

        cfg.validate_for_command(command)

        cfg.__post_init__()
        return cfg