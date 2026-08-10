import argparse
import json
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from cryoseed._version import __version__
from cryoseed.config import MainConfig
from cryoseed.config.config import (
    AbInitioConfig,
    DataConfig,
    HeteroRefineConfig,
    HomoRefineConfig,
    IOConfig,
    LoggingConfig,
    ModulesConfig,
    ReproduceConfig,
)
from cryoseed.utils import particle_mask as particle_mask_utils


class _HelpFormatter(argparse.RawTextHelpFormatter):
    pass


_PARTICLE_MASK_PROTECTION_PRESET_HELP = ", ".join(
    f"{coverage} -> {factor:.3f}"
    for coverage, factor in particle_mask_utils.PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS.items()
)

_FIELD_HELP_BY_PATH: dict[tuple[str, ...], str] = {
    ("io", "star_path"): "Input STAR file path.",
    ("io", "data_path"): "Optional root directory for particle image files.",
    ("io", "ref_volume_path"): "Optional input reference volume path.",
    ("io", "output_path"): "Output directory.",
    ("io", "ssd_cache_root"): (
        "SSD projection cache root for pose search "
        "(effective default: output_path/ssd_cache)."
    ),
    ("data", "batch_size"): "Dataloader batch size per process.",
    ("data", "num_workers"): "Number of dataloader worker processes.",
    ("data", "num_particles"): "Optional cap on number of particles (<=0 means all).",
    ("data", "image_size"): (
        "Target image side length D in pixels (<=0 means infer from data)."
    ),
    ("data", "angpix"): "Pixel size in Angstrom/pixel; overrides STAR if > 0.",
    ("data", "particle_diameter"): (
        "Particle diameter in Angstrom; required for solvent-mask sphere mode "
        "and particle-mask logic."
    ),
    ("logging", "log_dir"): "Log directory (effective default: output_path/logs).",
    ("logging", "log_prefix"): "Log filename prefix.",
    ("logging", "level"): "Root log level.",
    ("reproduce", "seed"): "Random seed.",
    ("reproduce", "deterministic"): (
        "Enable deterministic mode; may reduce performance."
    ),
    ("modules", "volume", "num_volumes"): (
        "Number of volumes/classes K (homorefine requires K=1)."
    ),
    ("modules", "volume", "voxel", "backproject_chunk"): (
        "Chunk size over poses in backprojection (memory/speed tradeoff)."
    ),
    ("modules", "volume", "voxel", "full_backprojection"): (
        "Use the full Fourier image radius during backprojection instead of the "
        "current side_length-limited radius."
    ),
    ("modules", "volume", "voxel", "learning_rate"): (
        "Global SGD learning rate for ab initio volume updates."
    ),
    ("modules", "volume", "voxel", "learning_rate_decay"): (
        "Exponential learning-rate decay factor applied after each solver update."
    ),
    ("modules", "volume", "voxel", "momentum"): (
        "SGD momentum used for ab initio volume updates."
    ),
    ("modules", "search", "init_healpix_order"): "Initial HEALPix order.",
    ("modules", "search", "neighbor_steps"): (
        "Local neighborhood radius in grid steps."
    ),
    ("modules", "search", "init_trans_grid_extent"): (
        "Initial translation search extent in pixels; null means auto = "
        "data.image_size // 2."
    ),
    ("modules", "search", "trans_grid_samples"): (
        "Base number of translation-grid samples per axis."
    ),
    ("modules", "search", "trans_grid_x_shift"): (
        "Translation grid x offset in pixels."
    ),
    ("modules", "search", "trans_grid_y_shift"): (
        "Translation grid y offset in pixels."
    ),
    ("modules", "search", "pose_chunk_factor"): (
        "Chunking factor for projection/translation computation "
        "(memory/speed tradeoff)."
    ),
    ("modules", "search", "max_candidates"): (
        "Max number of pose candidates kept per image; use -1 for unlimited."
    ),
    ("modules", "search", "criterion_chunk"): (
        "Chunk size for pose-search criterion evaluation."
    ),
    ("modules", "search", "candidate_select_threshold"): (
        "Cumulative probability threshold for candidate selection."
    ),
    ("modules", "search", "volume_class_similarity"): (
        "Mix each image's posterior marginal over the K-way volume-class axis "
        "toward the uniform distribution before reconstructing the joint posterior."
    ),
    ("modules", "search", "volume_class_similarity_scope"): (
        "Scope for applying volume_class_similarity: 'global' applies only during "
        "global pose search, and 'all' applies during both global and local search."
    ),
    ("modules", "search", "oversampling_deduplicate"): (
        "Deduplicate identical oversampling hypotheses before evaluation."
    ),
    ("modules", "search", "ring_averaged_mse"): (
        "Average MSE contributions within each Fourier ring instead of summing them."
    ),
    ("modules", "search", "particle_mask", "enabled"): (
        "Enable particle masking during preprocessing."
    ),
    ("modules", "search", "particle_mask", "zero_mask"): (
        "Fill masked-out real-space pixels with zeros instead of sampled noise."
    ),
    ("modules", "search", "particle_mask", "soft_edge_pixels"): (
        "Soft-edge width in pixels for the particle mask."
    ),
    ("modules", "search", "particle_mask", "protection_disable_epochs"): (
        "Disable particle masking for the first N refinement epochs as a warm-up."
    ),
    ("modules", "search", "particle_mask", "protection_radius_factor"): (
        "Scale factor that converts trans_update_rms into extra particle-mask radius."
    ),
    ("modules", "statistics", "noise", "enabled"): (
        "Enable noise variance spectrum estimation."
    ),
    ("modules", "statistics", "noise", "accumulate_chunk"): (
        "Chunk size over poses in noise accumulation (memory/speed tradeoff)."
    ),
    ("modules", "statistics", "noise", "init_variance"): (
        "Initial value for the noise variance spectrum."
    ),
    ("modules", "statistics", "noise", "precision_eps"): (
        "Clamp epsilon used when forming precision = 1 / variance."
    ),
    ("modules", "statistics", "noise", "ema_decay"): (
        "EMA decay used by noise variance running statistics."
    ),
    ("modules", "statistics", "noise", "prior_weight"): (
        "Weight of the fixed prior in noise variance regularization."
    ),
    ("modules", "statistics", "noise", "inflated_weight"): (
        "Initial weight of the inflated prior in noise variance regularization."
    ),
    ("modules", "statistics", "noise", "inflated_decay"): (
        "Decay applied to the inflated prior weight; null falls back to ema_decay."
    ),
    ("modules", "statistics", "noise", "inflated_scale"): (
        "Multiplicative scale applied to the inflated prior variance."
    ),
    ("modules", "statistics", "prior", "enabled"): (
        "Enable prior variance spectrum regularization."
    ),
    ("modules", "statistics", "prior", "init_variance"): (
        "Initial value for the prior variance spectrum."
    ),
    ("modules", "statistics", "prior", "precision_eps"): (
        "Clamp epsilon used when forming prior precision."
    ),
    ("modules", "statistics", "prior", "tail_floor"): (
        "Floor value for the high-frequency exponential tail of the prior."
    ),
    ("modules", "statistics", "prior", "init_lowpass_cutoff"): (
        "Low-pass size L in pixels used to initialize the prior tail."
    ),
    ("abinitio", "engine", "num_epochs"): "Number of ab initio epochs.",
    ("abinitio", "engine", "init_particles_per_volume"): (
        "Number of particles used to initialize each volume/class."
    ),
    ("abinitio", "engine", "init_lowpass_angstrom"): (
        "Initial low-pass resolution in Angstrom for the starting side_length."
    ),
    ("abinitio", "engine", "solvent_mask"): (
        "Solvent mask selector for ab initio: 'none', 'sphere', 'auto', "
        "or a mask file path."
    ),
    ("abinitio", "engine", "solvent_mask_soft_edge_pixels"): (
        "Soft-edge width in voxels for the spherical solvent mask used in ab initio."
    ),
    ("abinitio", "engine", "loss_ema_decay"): (
        "EMA decay used for the ab initio loss metric."
    ),
    ("abinitio", "engine", "pose_rms_ema_decay"): (
        "EMA decay used for the ab initio pose-update RMS metrics."
    ),
    ("abinitio", "scheduler", "schedule_check_interval_iters"): (
        "Requested iterations between ab initio scheduler checkpoints."
    ),
    ("abinitio", "scheduler", "confidence_threshold"): (
        "avg_confidence threshold for aggressive side_length growth."
    ),
    ("abinitio", "scheduler", "convergence_patience"): (
        "Number of consecutive checks required before declaring convergence."
    ),
    ("abinitio", "scheduler", "pose_translation_center_mode"): (
        "Translation-center mode for pose search: 'auto' lets the scheduler decide, "
        "'always' centers on stored pose translations, and 'never' starts from zero."
    ),
    ("abinitio", "scheduler", "increase_radius_step"): (
        "Default radius increment in frequency marching, in pixels."
    ),
    ("abinitio", "scheduler", "auto_local_healpix_order"): (
        "HEALPix order at which auto-local switches to Euler search."
    ),
    ("abinitio", "scheduler", "auto_local_assignment_change_threshold"): (
        "Maximum assignment-change rate allowed for the ab initio auto-local switch."
    ),
    ("abinitio", "scheduler", "trans_extent_scale"): (
        "Update trans_grid_extent to this factor times trans_update_rms."
    ),
    ("abinitio", "scheduler", "target_side_length_resolution"): (
        "Target side-length-derived resolution used to stop side_length growth."
    ),
    ("abinitio", "scheduler", "target_healpix_order"): (
        "Hard upper bound for the ab initio base HEALPix order."
    ),
    ("abinitio", "scheduler", "pose_rotation_stability_factor"): (
        "Scale factor applied to the current HEALPix angular step when testing rotational pose stability."
    ),
    ("abinitio", "scheduler", "pose_translation_stability_factor"): (
        "Scale factor applied to the current translation-grid spacing when testing translational pose stability."
    ),
    ("heterorefine", "scheduler", "pose_translation_center_mode"): (
        "Translation-center mode for pose search: 'auto' and 'always' center on "
        "stored pose translations, while 'never' starts from zero."
    ),
    ("homorefine", "engine", "num_epochs"): (
        "Number of homogeneous-refinement epochs."
    ),
    ("homorefine", "engine", "init_lowpass_angstrom"): (
        "Initial low-pass resolution in Angstrom for setting side_length."
    ),
    ("homorefine", "engine", "external_reconstruct"): (
        "Enable external reconstruction data export under output_path/external_reconstruct."
    ),
    ("homorefine", "engine", "solvent_mask"): (
        "Solvent mask selector for homorefine: 'none', 'sphere', 'auto', "
        "or a mask file path."
    ),
    ("homorefine", "engine", "solvent_mask_soft_edge_pixels"): (
        "Soft-edge width in voxels for the spherical solvent mask."
    ),
    ("homorefine", "engine", "solvent_fsc_correction"): (
        "Enable solvent-mask-aware FSC correction during homorefine."
    ),
    ("homorefine", "scheduler", "first_epoch_ncc"): (
        "Use an NCC-based correlation criterion on the first homorefine epoch only."
    ),
    ("homorefine", "scheduler", "confidence_threshold"): (
        "avg_confidence threshold for aggressive side_length growth."
    ),
    ("homorefine", "scheduler", "convergence_patience"): (
        "Number of consecutive epochs required before declaring convergence."
    ),
    ("homorefine", "scheduler", "fsc_resolution_improvement_threshold"): (
        "Minimum FSC-resolution improvement required to reset the no-gain counter."
    ),
    ("homorefine", "scheduler", "fsc_resolution_rebound_threshold"): (
        "Maximum FSC-resolution rebound still treated as no meaningful gain."
    ),
    ("homorefine", "scheduler", "trans_update_rms_threshold"): (
        "Maximum translation-update RMS treated as a small pose update for convergence."
    ),
    ("homorefine", "scheduler", "pose_translation_center_mode"): (
        "Translation-center mode for pose search: 'auto', 'always', or 'never'."
    ),
    ("homorefine", "scheduler", "increase_radius_step"): (
        "Default radius increment in frequency marching, in pixels."
    ),
    ("homorefine", "scheduler", "increase_radius_aggressive_factor"): (
        "Extra radius increment factor when confidence is high."
    ),
    ("homorefine", "scheduler", "increase_radius_aggressive_fsc_threshold"): (
        "Minimum FSC at the current side_length limit required to use aggressive radius growth."
    ),
    ("homorefine", "scheduler", "base_healpix_order"): (
        "Starting HEALPix order used while the scheduler stays in global search."
    ),
    ("homorefine", "scheduler", "auto_local_healpix_order"): (
        "HEALPix order at which auto-local switches from global HEALPix to local Euler search."
    ),
    ("homorefine", "scheduler", "use_cache"): (
        "Enable projection cache for pose search."
    ),
    ("homorefine", "scheduler", "cache_max_healpix_order"): (
        "Enable caching only when healpix_order <= this value."
    ),
    ("homorefine", "scheduler", "ssd_cache_min_side_length"): (
        "Use SSD cache when side_length >= this value; otherwise memory cache."
    ),
    ("homorefine", "scheduler", "trans_extent_scale"): (
        "Update trans_grid_extent to this factor times trans_update_rms."
    ),
}

_FIELD_DEFAULT_OVERRIDE_BY_PATH: dict[tuple[str, ...], str] = {
    ("io", "ssd_cache_root"): "output_path/ssd_cache",
    ("logging", "log_dir"): "output_path/logs",
}

_CLI_ARG_KWARGS_BY_PATH: dict[tuple[str, ...], dict[str, Any]] = {
    ("logging", "level"): {
        "choices": ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    },
    ("modules", "search", "volume_class_similarity_scope"): {
        "choices": ("global", "all"),
    },
    ("abinitio", "scheduler", "pose_translation_center_mode"): {
        "choices": ("auto", "always", "never"),
    },
    ("heterorefine", "scheduler", "pose_translation_center_mode"): {
        "choices": ("auto", "always", "never"),
    },
    ("homorefine", "scheduler", "pose_translation_center_mode"): {
        "choices": ("auto", "always", "never"),
    },
}


def _format_default_for_help(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if value else '""'
    return json.dumps(value)


def _field_default_repr(spec: dict[str, Any]) -> str:
    override = _FIELD_DEFAULT_OVERRIDE_BY_PATH.get(spec["path"])
    if override is not None:
        return override
    dc_field = spec["field"]
    if dc_field.default is not MISSING:
        return _format_default_for_help(dc_field.default)
    default_factory = getattr(dc_field, "default_factory", MISSING)
    if default_factory is not MISSING:
        if default_factory is dict:
            return "{}"
        if default_factory is list:
            return "[]"
        name = getattr(default_factory, "__name__", repr(default_factory))
        return f"<factory:{name}>"
    return "<required>"


def _build_leaf_help(spec: dict[str, Any]) -> str:
    summary = _FIELD_HELP_BY_PATH.get(spec["path"])
    if summary is None:
        summary = f"Override `{'.'.join(spec['path'])}`."
    return f"{summary} Default: {_field_default_repr(spec)}."


from cryoseed.engines.abinitio import AbInitioEngine
from cryoseed.engines.heterorefine import HeteroRefineEngine
from cryoseed.engines.homorefine import HomoRefineEngine
from cryoseed.runtime.distributed import cleanup_runtime, setup_runtime


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is None:
        return tp
    args = [arg for arg in get_args(tp) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return tp


def _leaf_field_specs(root_name: str, cls: type) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def _walk(current_cls: type, path: tuple[str, ...], logical: tuple[str, ...]) -> None:
        type_hints = get_type_hints(current_cls)
        for dc_field in fields(current_cls):
            field_type = _unwrap_optional(type_hints[dc_field.name])
            next_path = path + (dc_field.name,)
            next_logical = logical + (dc_field.name,)
            if is_dataclass(field_type):
                _walk(field_type, next_path, next_logical)
            elif field_type is dict:
                continue
            else:
                specs.append(
                    {
                        "path": next_path,
                        "logical": next_logical,
                        "leaf": dc_field.name,
                        "type": field_type,
                        "field": dc_field,
                    }
                )

    _walk(cls, (root_name,), ())
    return specs


def _flag_name(spec: dict[str, Any], leaf_counts: dict[str, int]) -> str:
    if leaf_counts[spec["leaf"]] == 1:
        return spec["leaf"].replace("_", "-")
    parts = list(spec["logical"])
    for start in range(len(parts) - 2, -1, -1):
        candidate = "-".join(parts[start:]).replace("_", "-")
        if candidate:
            return candidate
    return spec["leaf"].replace("_", "-")


def _add_leaf_argument(
    group: argparse._ArgumentGroup,
    *,
    spec: dict[str, Any],
    flag_name: str,
    dest_map: dict[str, tuple[str, ...]],
    aliases: list[str] | None = None,
) -> None:
    dest = "__".join(spec["path"])
    option_strings = [f"--{flag_name}"]
    if aliases:
        option_strings.extend(aliases)
    kwargs: dict[str, Any] = {
        "dest": dest,
        "default": None,
    }
    field_type = _unwrap_optional(spec["type"])
    if field_type is bool:
        kwargs["action"] = argparse.BooleanOptionalAction
    elif field_type in (int, float, str):
        kwargs["type"] = field_type
    else:
        kwargs["type"] = str
    kwargs.update(_CLI_ARG_KWARGS_BY_PATH.get(spec["path"], {}))
    kwargs.setdefault("help", _build_leaf_help(spec))
    group.add_argument(*option_strings, **kwargs)
    dest_map[dest] = spec["path"]


def _add_nested_config_overrides(
    parser: argparse.ArgumentParser,
    *,
    command: str,
) -> None:
    shared_sections = [
        ("io", IOConfig),
        ("data", DataConfig),
        ("logging", LoggingConfig),
        ("reproduce", ReproduceConfig),
        ("modules", ModulesConfig),
    ]
    command_sections = {
        "abinitio": ("abinitio", AbInitioConfig),
        "heterorefine": ("heterorefine", HeteroRefineConfig),
        "homorefine": ("homorefine", HomoRefineConfig),
    }
    specs: list[dict[str, Any]] = []
    for root_name, cls in shared_sections:
        specs.extend(_leaf_field_specs(root_name, cls))
    specs.extend(_leaf_field_specs(*command_sections[command]))
    if command != "abinitio":
        gradient_paths = {
            ("modules", "volume", "voxel", "learning_rate"),
            ("modules", "volume", "voxel", "learning_rate_decay"),
            ("modules", "volume", "voxel", "momentum"),
        }
        specs = [spec for spec in specs if spec["path"] not in gradient_paths]

    leaf_counts: dict[str, int] = {}
    for spec in specs:
        leaf_counts[spec["leaf"]] = leaf_counts.get(spec["leaf"], 0) + 1

    dest_map: dict[str, tuple[str, ...]] = {}
    aliases_by_flag = {
        "star-path": ["-i"],
        "output-path": ["-o"],
        "num-epochs": ["-n"],
        "batch-size": ["-b"],
        "num-workers": ["-w"],
        "level": ["--log-level"],
        "particle-mask-enabled": ["--use-particle-mask"],
        "noise-enabled": ["--use-noise"],
        "prior-enabled": ["--use-prior"],
    }

    groups: dict[str, argparse._ArgumentGroup] = {}
    for spec in specs:
        group_key = spec["path"][0]
        if len(spec["path"]) > 1 and spec["path"][0] in ("modules", command):
            group_key = ".".join(spec["path"][:2])
        group = groups.setdefault(group_key, parser.add_argument_group(group_key))
        flag_name = _flag_name(spec, leaf_counts)
        aliases = aliases_by_flag.get(flag_name)
        _add_leaf_argument(
            group,
            spec=spec,
            flag_name=flag_name,
            aliases=aliases,
            dest_map=dest_map,
        )

    parser.set_defaults(_config_dest_map=dest_map)

    particle_mask_group = parser.add_argument_group("particle mask")
    particle_mask_group.add_argument(
        "--zero-particle-mask",
        dest="modules__search__particle_mask__zero_mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fill masked-out real-space pixels with zeros instead of sampled noise. "
            "Use `--no-zero-particle-mask` to sample background noise."
        ),
    )
    particle_mask_group.add_argument(
        "--particle-mask-protection-coverage",
        dest="particle_mask_protection_coverage",
        choices=tuple(particle_mask_utils.PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS),
        default=None,
        help=(
            "Recommended particle-mask protection presets: "
            f"{_PARTICLE_MASK_PROTECTION_PRESET_HELP}."
        ),
    )

    ctf_group = parser.add_argument_group("ctf fallback overrides")
    ctf_group.add_argument(
        "--default-optic-params-json",
        dest="default_optic_params_json",
        default=None,
        help=(
            "Fallback optics-level CTF fields as a JSON object string."
        ),
    )
    ctf_group.add_argument(
        "--default-particle-params-json",
        dest="default_particle_params_json",
        default=None,
        help=(
            "Fallback particle-level CTF fields as a JSON object string."
        ),
    )
    ctf_group.add_argument(
        "--voltage-kv",
        dest="default_optic_params_voltage_kv",
        type=float,
        default=None,
        help="Fallback accelerating voltage in kV.",
    )
    ctf_group.add_argument(
        "--spherical-aberration-mm",
        dest="default_optic_params_spherical_aberration_mm",
        type=float,
        default=None,
        help="Fallback spherical aberration in mm.",
    )
    ctf_group.add_argument(
        "--ctf-bfactor",
        dest="default_optic_params_ctf_bfactor",
        type=float,
        default=None,
        help="Fallback CTF B-factor.",
    )
    ctf_group.add_argument(
        "--ctf-scale",
        dest="default_optic_params_ctf_scale",
        type=float,
        default=None,
        help="Fallback CTF scale factor.",
    )
    ctf_group.add_argument(
        "--amplitude-contrast",
        dest="default_optic_params_amplitude_contrast",
        type=float,
        default=None,
        help="Fallback amplitude contrast.",
    )
    ctf_group.add_argument(
        "--phase-shift-deg",
        dest="default_optic_params_phase_shift_deg",
        type=float,
        default=None,
        help="Fallback phase shift in degrees.",
    )
    ctf_group.add_argument(
        "--defocus-u-angstrom",
        dest="default_particle_params_defocus_u_angstrom",
        type=float,
        default=None,
        help="Fallback particle defocus U in Angstrom.",
    )
    ctf_group.add_argument(
        "--defocus-v-angstrom",
        dest="default_particle_params_defocus_v_angstrom",
        type=float,
        default=None,
        help="Fallback particle defocus V in Angstrom.",
    )
    ctf_group.add_argument(
        "--defocus-angle-deg",
        dest="default_particle_params_defocus_angle_deg",
        type=float,
        default=None,
        help="Fallback particle defocus angle in degrees.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryoseed",
        description="Cryo-EM reconstruction toolkit",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def _add_runtime_args(command_parser: argparse.ArgumentParser, *, command_name: str) -> None:
        command_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=argparse.SUPPRESS,
        help="Path to a YAML/JSON config file. Command-line arguments override config values, and unspecified keys fall back to command defaults.",
        )
        command_parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Data-parallel group size (dp). Must satisfy dp * cp == WORLD_SIZE.",
        )
        command_parser.add_argument(
        "--compute-parallel-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Compute-parallel group size (cp). Must satisfy dp * cp == WORLD_SIZE.",
        )
        resume_group = command_parser.add_mutually_exclusive_group()
        resume_group.add_argument(
        "--resume",
        type=str,
        default=argparse.SUPPRESS,
        help=f"Resume {command_name} from a checkpoint and continue from its next epoch.",
        )
        resume_group.add_argument(
        "--auto-resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Resume from output_path/checkpoints/latest.pt if it exists; otherwise start fresh.",
        )

    abinitio = subparsers.add_parser(
        "abinitio",
        help="Run ab initio reconstruction",
        description=(
            "Run ab initio reconstruction.\n\n"
            "Precedence: CLI options > user config file > command defaults > built-in defaults.\n"
            "The config file is optional.\n"
            "Options not provided on the command line fall back to the user config file,\n"
            "then to command defaults, and finally to built-in defaults."
        ),
        formatter_class=_HelpFormatter,
    )
    _add_runtime_args(abinitio, command_name="abinitio")
    _add_nested_config_overrides(abinitio, command="abinitio")

    heterorefine = subparsers.add_parser(
        "heterorefine",
        help="Run heterogeneous refinement",
        description=(
            "Run heterogeneous refinement.\n\n"
            "Precedence: CLI options > user config file > command defaults > built-in defaults."
        ),
        formatter_class=_HelpFormatter,
    )
    _add_runtime_args(heterorefine, command_name="heterorefine")
    _add_nested_config_overrides(heterorefine, command="heterorefine")

    homorefine = subparsers.add_parser(
        "homorefine",
        help="Run homogeneous refinement",
        description=(
            "Run homogeneous refinement.\n\n"
            "Precedence: CLI options > user config file > command defaults > built-in defaults.\n"
            "The config file is optional.\n"
            "Options not provided on the command line fall back to the user config file,\n"
            "then to command defaults, and finally to built-in defaults."
        ),
        formatter_class=_HelpFormatter,
    )
    _add_runtime_args(homorefine, command_name="homorefine")
    _add_nested_config_overrides(homorefine, command="homorefine")

    return parser


def run_abinitio(args) -> None:
    config = MainConfig.from_cli_args(args)

    runtime = setup_runtime(
        data_parallel_size=getattr(args, "data_parallel_size", None),
        compute_parallel_size=getattr(args, "compute_parallel_size", None),
    )
    try:
        if runtime.rank == 0:
            saved_config_path, snapshot_config_path = config.save_output_config(
                command="abinitio"
            )
            print(
                f"Saved launch config to {saved_config_path} and timestamped snapshot to {snapshot_config_path}",
                flush=True,
            )

        engine = AbInitioEngine(
            config=config,
            runtime=runtime,
            resume_checkpoint_path=getattr(args, "resume", None),
            auto_resume=bool(getattr(args, "auto_resume", False)),
        )
        engine.run()
    finally:
        cleanup_runtime()


def run_heterorefine(args) -> None:
    config = MainConfig.from_cli_args(args)
    runtime = setup_runtime(
        data_parallel_size=getattr(args, "data_parallel_size", None),
        compute_parallel_size=getattr(args, "compute_parallel_size", None),
    )
    try:
        if runtime.rank == 0:
            saved_config_path, snapshot_config_path = config.save_output_config(
                command="heterorefine"
            )
            print(
                f"Saved launch config to {saved_config_path} and timestamped snapshot to {snapshot_config_path}",
                flush=True,
            )

        engine = HeteroRefineEngine(
            config=config,
            runtime=runtime,
            resume_checkpoint_path=getattr(args, "resume", None),
            auto_resume=bool(getattr(args, "auto_resume", False)),
        )
        engine.run()
    finally:
        cleanup_runtime()


def run_homorefine(args) -> None:
    config = MainConfig.from_cli_args(args)

    runtime = setup_runtime(
        data_parallel_size=getattr(args, "data_parallel_size", None),
        compute_parallel_size=getattr(args, "compute_parallel_size", None),
    )
    try:
        if runtime.rank == 0:
            saved_config_path, snapshot_config_path = config.save_output_config(
                command="homorefine"
            )
            print(
                f"Saved launch config to {saved_config_path} and timestamped snapshot to {snapshot_config_path}",
                flush=True,
            )
        engine = HomoRefineEngine(
            config=config,
            runtime=runtime,
            resume_checkpoint_path=getattr(args, "resume", None),
            auto_resume=bool(getattr(args, "auto_resume", False)),
        )
        engine.run()
    finally:
        cleanup_runtime()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "abinitio":
        run_abinitio(args)
    elif args.command == "heterorefine":
        run_heterorefine(args)
    elif args.command == "homorefine":
        run_homorefine(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()