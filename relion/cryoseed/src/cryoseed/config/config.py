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

    return {key: normalized.get(key) for key in preferred_keys}


@dataclass
class IOConfig:
    star_path: str = ""
    data_path: str = ""
    ref_volume_path: str = ""
    output_path: str = "outputs"
    ssd_cache_root: str = ""


@dataclass
class DataConfig:
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
    log_dir: str = ""
    log_prefix: str = "cryoseed"
    level: str = "INFO"


@dataclass
class ReproduceConfig:
    seed: int = 42
    deterministic: bool = False


@dataclass
class ParticleMaskConfig:
    enabled: bool = False
    zero_mask: bool = True
    soft_edge_pixels: float = 5.0
    protection_disable_epochs: int = 5
    protection_radius_factor: float = (
        particle_mask_utils.DEFAULT_PARTICLE_MASK_PROTECTION_RADIUS_FACTOR
    )


@dataclass
class VolumeModuleConfig:
    num_volumes: int = 1
    backproject_chunk: int = 16384
    full_backprojection: bool = False


@dataclass
class SearchModuleConfig:
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
    particle_mask: ParticleMaskConfig = field(default_factory=ParticleMaskConfig)


@dataclass
class NoiseStatisticsConfig:
    enabled: bool = True
    accumulate_chunk: int = 16384
    init_variance: float = 1.0
    precision_eps: float = 1e-6
    ema_decay: float = 0.0
    prior_weight: float = 0.0
    inflated_weight: float = 0.0
    inflated_decay: float | None = None
    inflated_scale: float = 8.0


@dataclass
class PriorStatisticsConfig:
    enabled: bool = True
    init_variance: float = 1.0
    precision_eps: float = 1e-6
    tail_floor: float = 1e-5
    init_lowpass_cutoff: int | None = None


@dataclass
class StatisticsModuleConfig:
    noise: NoiseStatisticsConfig = field(default_factory=NoiseStatisticsConfig)
    prior: PriorStatisticsConfig = field(default_factory=PriorStatisticsConfig)


@dataclass
class ModulesConfig:
    volume: VolumeModuleConfig = field(default_factory=VolumeModuleConfig)
    search: SearchModuleConfig = field(default_factory=SearchModuleConfig)
    statistics: StatisticsModuleConfig = field(default_factory=StatisticsModuleConfig)


@dataclass
class AbInitioEngineConfig:
    num_epochs: int = 1000
    init_particles_per_volume: int = 100
    init_lowpass_angstrom: float = 50
    solvent_mask: str = "none"
    solvent_mask_soft_edge_pixels: float = 5.0
    loss_ema_decay: float = 0.9
    pose_rms_ema_decay: float = 0.9


@dataclass
class AbInitioSolverConfig:
    learning_rate: float = 1.0
    learning_rate_decay: float = 0.9995
    momentum: float = 0.9


@dataclass
class AbInitioSchedulerConfig:
    schedule_check_interval_iters: int = 100
    confidence_threshold: float = 0.5
    convergence_patience: int = 3
    pose_translation_center_mode: str = "auto"
    increase_radius_step: int = 1
    auto_local_healpix_order: int = 4
    auto_local_assignment_change_threshold: float = 1.0
    trans_extent_scale: float = 3.0
    target_side_length_resolution: float = 10.0
    target_healpix_order: int | None = None
    pose_rotation_stability_factor: float = 1.0
    pose_translation_stability_factor: float = 0.5


@dataclass
class AbInitioConfig:
    engine: AbInitioEngineConfig = field(default_factory=AbInitioEngineConfig)
    solver: AbInitioSolverConfig = field(default_factory=AbInitioSolverConfig)
    scheduler: AbInitioSchedulerConfig = field(default_factory=AbInitioSchedulerConfig)


@dataclass
class HomoRefineEngineConfig:
    num_epochs: int = 50
    init_lowpass_angstrom: float = 30
    external_reconstruct: bool = False
    solvent_mask: str = "none"
    solvent_mask_soft_edge_pixels: float = 5.0
    solvent_fsc_correction: bool = False


@dataclass
class HomoRefineSchedulerConfig:
    first_epoch_ncc: bool = True
    confidence_threshold: float = 0.1
    convergence_patience: int = 3
    fsc_resolution_improvement_threshold: float = 5e-3
    fsc_resolution_rebound_threshold: float = 1e-2
    trans_update_rms_threshold: float = 0.5
    pose_translation_center_mode: str = "auto"
    increase_radius_step: int = 10
    increase_radius_aggressive_factor: float = 0.25
    increase_radius_aggressive_fsc_threshold: float = 0.2
    base_healpix_order: int = 3
    auto_local_healpix_order: int = 4
    use_cache: bool = False
    cache_max_healpix_order: int = 4
    ssd_cache_min_side_length: int = 150
    trans_extent_scale: float = 3.0


@dataclass
class HomoRefineConfig:
    engine: HomoRefineEngineConfig = field(default_factory=HomoRefineEngineConfig)
    scheduler: HomoRefineSchedulerConfig = field(default_factory=HomoRefineSchedulerConfig)


@dataclass
class MainConfig:
    io: IOConfig = field(default_factory=IOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reproduce: ReproduceConfig = field(default_factory=ReproduceConfig)
    modules: ModulesConfig = field(default_factory=ModulesConfig)
    abinitio: AbInitioConfig = field(default_factory=AbInitioConfig)
    homorefine: HomoRefineConfig = field(default_factory=HomoRefineConfig)

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
            if value.strip() != value or any(
                ch in value for ch in ':\'"#{}[]&,*!?|<>=%@`\\\n\r\t'
            ):
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
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(cls._to_yaml_lines(item, indent=indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {cls._yaml_scalar_repr(item)}")
            return lines
        if isinstance(value, list):
            if not value:
                return [f"{prefix}[]"]
            lines: list[str] = []
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
        data = self.to_dict()
        if not command:
            return data
        if command not in ("abinitio", "homorefine"):
            raise ValueError(f"Unknown command: {command}")
        return {
            "io": data["io"],
            "data": data["data"],
            "logging": data["logging"],
            "reproduce": data["reproduce"],
            "modules": data["modules"],
            command: data[command],
        }

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
    def _load_command_defaults(cls, command: str | None) -> dict[str, Any]:
        if not command:
            return {}
        path = Path(__file__).resolve().parent / "defaults" / f"{command}.yaml"
        if not path.is_file():
            return {}
        return cls._load_file(str(path))

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
    def _validate_keys(cls, data: Any, schema: Any, *, path: str = "") -> dict[str, Any]:
        if data is None:
            return {}
        if not isinstance(data, dict):
            location = f" `{path}`" if path else ""
            raise TypeError(f"config section{location} must be a dict, got {type(data)}")
        allowed = {f.name for f in fields(schema)}
        normalized = dict(data)
        for key in normalized:
            if key not in allowed:
                location = f"{path}.{key}" if path else str(key)
                raise ValueError(f"Unknown config key `{location}`.")
        return normalized

    @classmethod
    def _build_io(cls, data: Any) -> IOConfig:
        return IOConfig(**cls._validate_keys(data, IOConfig, path="io"))

    @classmethod
    def _build_data(cls, data: Any) -> DataConfig:
        normalized = cls._validate_keys(data, DataConfig, path="data")
        normalized["default_optic_params"] = _normalize_param_keys(
            normalized.get("default_optic_params"),
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
        normalized["default_particle_params"] = _normalize_param_keys(
            normalized.get("default_particle_params"),
            [
                "defocus_u_angstrom",
                "defocus_v_angstrom",
                "defocus_angle_deg",
            ],
            location="data.default_particle_params",
        )
        return DataConfig(**normalized)

    @classmethod
    def _build_logging(cls, data: Any) -> LoggingConfig:
        return LoggingConfig(**cls._validate_keys(data, LoggingConfig, path="logging"))

    @classmethod
    def _build_reproduce(cls, data: Any) -> ReproduceConfig:
        return ReproduceConfig(**cls._validate_keys(data, ReproduceConfig, path="reproduce"))

    @classmethod
    def _build_particle_mask(cls, data: Any) -> ParticleMaskConfig:
        return ParticleMaskConfig(
            **cls._validate_keys(data, ParticleMaskConfig, path="modules.search.particle_mask")
        )

    @classmethod
    def _build_volume_module(cls, data: Any) -> VolumeModuleConfig:
        return VolumeModuleConfig(
            **cls._validate_keys(data, VolumeModuleConfig, path="modules.volume")
        )

    @classmethod
    def _build_search_module(cls, data: Any) -> SearchModuleConfig:
        normalized = cls._validate_keys(data, SearchModuleConfig, path="modules.search")
        normalized["particle_mask"] = cls._build_particle_mask(normalized.get("particle_mask"))
        return SearchModuleConfig(**normalized)

    @classmethod
    def _build_noise_statistics(cls, data: Any) -> NoiseStatisticsConfig:
        return NoiseStatisticsConfig(
            **cls._validate_keys(data, NoiseStatisticsConfig, path="modules.statistics.noise")
        )

    @classmethod
    def _build_prior_statistics(cls, data: Any) -> PriorStatisticsConfig:
        return PriorStatisticsConfig(
            **cls._validate_keys(data, PriorStatisticsConfig, path="modules.statistics.prior")
        )

    @classmethod
    def _build_statistics_module(cls, data: Any) -> StatisticsModuleConfig:
        normalized = cls._validate_keys(data, StatisticsModuleConfig, path="modules.statistics")
        normalized["noise"] = cls._build_noise_statistics(normalized.get("noise"))
        normalized["prior"] = cls._build_prior_statistics(normalized.get("prior"))
        return StatisticsModuleConfig(**normalized)

    @classmethod
    def _build_modules(cls, data: Any) -> ModulesConfig:
        normalized = cls._validate_keys(data, ModulesConfig, path="modules")
        normalized["volume"] = cls._build_volume_module(normalized.get("volume"))
        normalized["search"] = cls._build_search_module(normalized.get("search"))
        normalized["statistics"] = cls._build_statistics_module(normalized.get("statistics"))
        return ModulesConfig(**normalized)

    @classmethod
    def _build_abinitio(cls, data: Any) -> AbInitioConfig:
        normalized = cls._validate_keys(data, AbInitioConfig, path="abinitio")
        normalized["engine"] = AbInitioEngineConfig(
            **cls._validate_keys(normalized.get("engine"), AbInitioEngineConfig, path="abinitio.engine")
        )
        normalized["solver"] = AbInitioSolverConfig(
            **cls._validate_keys(normalized.get("solver"), AbInitioSolverConfig, path="abinitio.solver")
        )
        normalized["scheduler"] = AbInitioSchedulerConfig(
            **cls._validate_keys(
                normalized.get("scheduler"),
                AbInitioSchedulerConfig,
                path="abinitio.scheduler",
            )
        )
        return AbInitioConfig(**normalized)

    @classmethod
    def _build_homorefine(cls, data: Any) -> HomoRefineConfig:
        normalized = cls._validate_keys(data, HomoRefineConfig, path="homorefine")
        normalized["engine"] = HomoRefineEngineConfig(
            **cls._validate_keys(
                normalized.get("engine"),
                HomoRefineEngineConfig,
                path="homorefine.engine",
            )
        )
        normalized["scheduler"] = HomoRefineSchedulerConfig(
            **cls._validate_keys(
                normalized.get("scheduler"),
                HomoRefineSchedulerConfig,
                path="homorefine.scheduler",
            )
        )
        return HomoRefineConfig(**normalized)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: str | None = None) -> "MainConfig":
        if not isinstance(data, dict):
            raise TypeError(f"config root must be a dict, got {type(data)}")
        allowed = {f.name for f in fields(cls)}
        normalized = dict(data)
        for key in normalized:
            if key not in allowed:
                raise ValueError(f"Unknown config section `{key}`.")
        cfg = cls(
            io=cls._build_io(normalized.get("io")),
            data=cls._build_data(normalized.get("data")),
            logging=cls._build_logging(normalized.get("logging")),
            reproduce=cls._build_reproduce(normalized.get("reproduce")),
            modules=cls._build_modules(normalized.get("modules")),
            abinitio=cls._build_abinitio(normalized.get("abinitio")),
            homorefine=cls._build_homorefine(normalized.get("homorefine")),
        )
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
            cfg.abinitio.engine.solvent_mask = cls._resolve_solvent_mask_path(
                cfg.abinitio.engine.solvent_mask,
                base,
            )
            cfg.homorefine.engine.solvent_mask = cls._resolve_solvent_mask_path(
                cfg.homorefine.engine.solvent_mask,
                base,
            )
        return cfg

    @classmethod
    def from_file(cls, path: str, *, command: str | None = None) -> "MainConfig":
        del command
        data = cls._load_file(path)
        return cls.from_dict(data, base_dir=str(Path(path).resolve().parent))

    def validate_for_command(self, command: str | None) -> "MainConfig":
        del command
        return self

    @staticmethod
    def _set_nested_value(root: Any, path: tuple[str, ...], value: Any) -> None:
        target = root
        for name in path[:-1]:
            target = getattr(target, name)
        setattr(target, path[-1], value)

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

        dest_map = getattr(args, "_config_dest_map", {})
        for dest, path in dest_map.items():
            if not hasattr(args, dest):
                continue
            value = getattr(args, dest)
            if value is None:
                continue
            cls._set_nested_value(cfg, path, value)

        if hasattr(args, "default_optic_params_json") and getattr(args, "default_optic_params_json", None) is not None:
            parsed = json.loads(getattr(args, "default_optic_params_json"))
            if not isinstance(parsed, dict):
                raise TypeError("default_optic_params_json must be a JSON object")
            cfg.data.default_optic_params = parsed
        if hasattr(args, "default_particle_params_json") and getattr(args, "default_particle_params_json", None) is not None:
            parsed = json.loads(getattr(args, "default_particle_params_json"))
            if not isinstance(parsed, dict):
                raise TypeError("default_particle_params_json must be a JSON object")
            cfg.data.default_particle_params = parsed

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
                value = getattr(args, arg_name)
                if value is not None:
                    cfg.data.default_optic_params[key] = float(value)

        particle_keys = ["defocus_u_angstrom", "defocus_v_angstrom", "defocus_angle_deg"]
        for key in particle_keys:
            arg_name = f"default_particle_params_{key}"
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    cfg.data.default_particle_params[key] = float(value)

        if hasattr(args, "particle_mask_protection_coverage"):
            value = getattr(args, "particle_mask_protection_coverage")
            if value is not None:
                cfg.modules.search.particle_mask.protection_radius_factor = (
                    particle_mask_utils.PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS[str(value)]
                )

        cfg.data.default_optic_params = _normalize_param_keys(
            cfg.data.default_optic_params,
            optic_keys,
            location="data.default_optic_params",
        )
        cfg.data.default_particle_params = _normalize_param_keys(
            cfg.data.default_particle_params,
            particle_keys,
            location="data.default_particle_params",
        )

        output_path_changed = (
            hasattr(args, "output_path")
            and getattr(args, "output_path", None) is not None
            and cfg.io.output_path != initial_output_path
        )
        if getattr(args, "log_dir", None) is None and (
            (not cfg.logging.log_dir)
            or (
                output_path_changed
                and cfg.logging.log_dir == str(Path(initial_output_path) / "logs")
            )
        ):
            cfg.logging.log_dir = str(Path(cfg.io.output_path) / "logs")
        if getattr(args, "ssd_cache_root", None) is None and (
            (not cfg.io.ssd_cache_root)
            or (
                output_path_changed
                and cfg.io.ssd_cache_root == str(Path(initial_output_path) / "ssd_cache")
            )
        ):
            cfg.io.ssd_cache_root = str(Path(cfg.io.output_path) / "ssd_cache")

        if base_dir is not None:
            base = Path(base_dir)
            cfg.abinitio.engine.solvent_mask = cls._resolve_solvent_mask_path(
                cfg.abinitio.engine.solvent_mask,
                base,
            )
            cfg.homorefine.engine.solvent_mask = cls._resolve_solvent_mask_path(
                cfg.homorefine.engine.solvent_mask,
                base,
            )

        cfg.__post_init__()
        return cfg