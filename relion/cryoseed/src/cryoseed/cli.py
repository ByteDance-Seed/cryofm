import argparse
import json
from dataclasses import MISSING, fields
from typing import Any, get_args, get_origin, get_type_hints

from cryoseed._version import __version__
from cryoseed.config import MainConfig
from cryoseed.config.config import COMMAND_HIDDEN_FIELDS
from cryoseed.utils import particle_mask as particle_mask_utils


class _HelpFormatter(argparse.RawTextHelpFormatter):
    pass


class _PairedBooleanAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        kwargs.setdefault("nargs", 0)
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, _parser, namespace, _values, option_string=None):
        del _parser, _values
        setattr(namespace, self.dest, not str(option_string).startswith("--no-"))


_PARTICLE_MASK_PROTECTION_PRESET_HELP = ", ".join(
    f"{coverage} -> {factor:.3f}"
    for coverage, factor in particle_mask_utils.PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS.items()
)


from cryoseed.engines.abinitio import AbInitioEngine
from cryoseed.engines.homorefine import HomoRefineEngine
from cryoseed.runtime.distributed import cleanup_runtime, setup_runtime


def _is_optional(tp: Any) -> tuple[bool, Any]:
    origin = get_origin(tp)
    if origin is None:
        return False, tp

    args = get_args(tp)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) < len(args):
        if len(non_none) == 1:
            return True, non_none[0]
        return True, tp
    return False, tp


def _looks_like_dict(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin is None:
        return tp is dict
    return origin is dict


def _format_default_for_help(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return repr(value)
    if value is None:
        return "None"
    return repr(value)


_FIELD_HELP: dict[tuple[str, str], str] = {
    ("io", "star_path"): "Input RELION STAR file (particles/poses/CTF).",
    ("io", "data_path"): "Prefix prepended to relative stack paths from STAR.",
    ("io", "ref_volume_path"): "Initial reference volume (MRC) used to initialize refinement.",
    ("io", "output_path"): "Output root directory (maps/fsc/logs/cache).",
    ("io", "ssd_cache_root"): "SSD projection cache root for pose search (default: output_path/ssd_cache).",

    ("data", "batch_size"): "Dataloader batch size (per process).",
    ("data", "num_workers"): "Number of dataloader worker processes.",
    ("data", "num_particles"): "Optional cap on number of particles (<=0 means all).",
    ("data", "image_size"): "Target image side length D in pixels (<=0 means infer from data).",
    ("data", "angpix"): "Pixel size in Angstrom/pixel; overrides STAR if > 0.",
    ("data", "particle_diameter"): "Particle diameter in Angstrom; required for frequency marching.",
    ("data", "default_optic_params"): (
        "Fallback optics-level CTF fields as a JSON object string, e.g. "
        "'{\"voltage_kv\":300,\"spherical_aberration_mm\":2.7,\"ctf_bfactor\":0,"
        "\"ctf_scale\":1,\"amplitude_contrast\":0.1,\"phase_shift_deg\":0}'."
    ),
    ("data", "default_particle_params"): (
        "Fallback particle-level CTF fields as a JSON object string, e.g. "
        "'{\"defocus_u_angstrom\":10000,\"defocus_v_angstrom\":10000,"
        "\"defocus_angle_deg\":0}'."
    ),

    ("logging", "log_dir"): "Log directory (default: output_path/logs).",
    ("logging", "log_prefix"): "Log filename prefix.",
    ("logging", "level"): "Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",

    ("reconstruction", "num_volumes"): "Number of volumes/classes K (homorefine requires K=1).",
    ("reconstruction", "external_reconstruct"): "Enable external reconstruction data export under output_path/external_reconstruct.",
    ("reconstruction", "full_backprojection"): "Use the full Fourier image radius during backprojection instead of the current side_length-limited radius.",
    ("reconstruction", "backproject_chunk"): "Chunk size over poses in backprojection (memory/speed tradeoff).",
    ("reconstruction", "accumulate_chunk"): "Chunk size over poses in noise accumulation (memory/speed tradeoff).",

    ("statistics", "use_noise"): "Enable noise variance spectrum estimation.",
    ("statistics", "use_prior"): "Enable prior variance spectrum regularization.",
    ("statistics", "init_variance"): "Initial value for noise/prior variance spectra.",
    ("statistics", "precision_eps"): "Clamp epsilon used when forming precision = 1/variance.",
    ("statistics", "noise_ema_decay"): "EMA decay used by noise variance running statistics.",
    ("statistics", "noise_prior_weight"): "Weight of the fixed prior in noise variance regularization.",
    ("statistics", "noise_inflated_weight"): "Initial weight of the inflated prior in noise variance regularization.",
    ("statistics", "noise_inflated_decay"): "Decay applied to the inflated prior weight; defaults to noise_ema_decay when omitted.",
    ("statistics", "noise_inflated_scale"): "Multiplicative scale applied to the inflated prior variance.",
    ("statistics", "tail_floor"): "Floor value for the high-frequency exponential tail of the prior.",
    ("statistics", "init_lowpass_cutoff"): "Low-pass size L (pixels) used to initialize the prior tail.",

    ("homorefine", "num_epochs"): "Number of homogeneous-refinement epochs.",
    ("homorefine", "first_epoch_ncc"): "Use an NCC-based correlation criterion on the first homorefine epoch only.",
    ("homorefine", "fsc_threshold"): "FSC threshold used to estimate resolution.",
    ("homorefine", "init_lowpass_angstrom"): "Initial low-pass resolution (Angstrom) for setting side_length.",
    ("homorefine", "solvent_mask"): "Solvent mask selector: `none`, `sphere`, `auto`, or a mask file path.",
    ("homorefine", "solvent_mask_soft_edge_pixels"): "Soft-edge width in voxels for the spherical solvent mask.",
    ("homorefine", "solvent_fsc_correction"): "Enable solvent-mask-aware FSC correction during homorefine.",

    ("scheduler", "confidence_threshold"): "avg_confidence threshold for aggressive side_length growth.",
    ("scheduler", "convergence_patience"): "Number of consecutive epochs required before declaring convergence once both FSC and translation-update conditions are satisfied.",
    ("scheduler", "fsc_resolution_improvement_threshold"): "Minimum FSC-resolution improvement (Angstrom) required to reset the no-gain counter.",
    ("scheduler", "fsc_resolution_rebound_threshold"): "Maximum FSC-resolution rebound (Angstrom) still treated as no meaningful gain.",
    ("scheduler", "trans_update_rms_threshold"): "Maximum translation-update RMS (pixels) treated as a small pose update for convergence.",
    ("scheduler", "pose_translation_center_mode"): "Translation-center mode for pose search: `auto` lets the scheduler decide, `always` always centers on stored pose translations, and `never` always starts from zero-centered translation candidates.",
    ("scheduler", "increase_radius_step"): "Default radius increment in frequency marching (pixels).",
    ("scheduler", "increase_radius_aggressive_factor"): "Extra radius increment factor when confident.",
    ("scheduler", "increase_radius_aggressive_fsc_threshold"): "Minimum FSC at the current side_length limit required to use the aggressive radius increment.",
    ("scheduler", "base_healpix_order"): (
        "Starting HEALPix order used while the scheduler stays in global HEALPix search."
    ),
    ("scheduler", "auto_local_healpix_order"): (
        "HEALPix order at which the auto-local switch changes global HEALPix search to local Euler search."
    ),
    ("scheduler", "auto_local_assignment_change_threshold"): (
        "Maximum assignment-change rate allowed for the auto-local switch in ab initio mode."
    ),
    ("scheduler", "use_cache"): "Enable projection cache (memory/SSD) for pose search.",
    ("scheduler", "cache_max_healpix_order"): "Enable caching only when healpix_order <= this value.",
    ("scheduler", "ssd_cache_min_side_length"): "Use SSD cache when side_length >= this value; otherwise memory cache.",
    ("scheduler", "trans_extent_scale"): "Update trans_grid_extent to this factor times trans_update_rms.",
    ("data", "particle_mask.protection_disable_epochs"): "Disable particle masking for the first N refinement epochs as a warm-up.",
    ("data", "particle_mask.protection_radius_factor"): (
        "Scale factor that converts trans_update_rms (pixels) into an extra "
        "particle-mask radius in pixels before it is doubled and added to the "
        "mask diameter. Recommended presets assume isotropic 2D Gaussian "
        "translation errors, so the radial coverage obeys "
        "`coverage = 1 - exp(-factor^2)`."
    ),

    ("pose_search", "init_healpix_order"): "Initial HEALPix order.",
    ("pose_search", "neighbor_steps"): "Local neighborhood radius in grid steps.",
    ("pose_search", "init_trans_grid_extent"): (
        "Initial translation search extent in pixels; YAML null means auto = "
        "data.image_size // 2."
    ),
    ("pose_search", "trans_grid_samples"): "Base number of translation-grid samples per axis.",
    ("pose_search", "trans_grid_x_shift"): "Translation grid x offset (pixels).",
    ("pose_search", "trans_grid_y_shift"): "Translation grid y offset (pixels).",
    ("pose_search", "pose_chunk_factor"): "Chunking factor for projection/translation computation (memory/speed tradeoff).",
    ("pose_search", "max_candidates"): "Max number of pose candidates kept per image; use -1 for unlimited.",
    ("pose_search", "criterion_chunk"): "Chunk size for pose-search criterion evaluation.",
    ("pose_search", "candidate_select_threshold"): "Cumulative probability threshold for candidate selection.",
    ("pose_search", "volume_class_similarity"): (
        "Mix each image's posterior marginal over the K-way volume-class axis toward the uniform distribution before reconstructing the joint posterior."
    ),
    ("pose_search", "volume_class_similarity_scope"): (
        "Scope for applying volume_class_similarity: `global` applies only during global pose search, and `all` applies during both global and local pose search."
    ),
    ("pose_search", "ring_averaged_mse"): "Average MSE contributions within each Fourier ring instead of summing them.",

    ("reproduce", "seed"): "Random seed.",
    ("reproduce", "deterministic"): "Enable deterministic mode (may reduce performance).",

    ("abinitio", "init_particles_per_volume"): (
        "Number of particles used to initialize each volume/class in ab initio mode."
    ),
    ("abinitio", "num_epochs"): "Number of ab initio epochs.",
    ("abinitio", "init_lowpass_angstrom"): (
        "Initial low-pass resolution (Angstrom) used to filter the ab initio init_volume and set the starting side_length."
    ),
    ("abinitio", "solvent_mask"): (
        "Solvent mask selector for ab initio: `none`, `sphere`, `auto`, or a mask file path."
    ),
    ("abinitio", "solvent_mask_soft_edge_pixels"): (
        "Soft-edge width in voxels for the spherical solvent mask used in ab initio."
    ),
    ("abinitio", "learning_rate"): (
        "Global SGD learning rate for ab initio volume updates."
    ),
    ("abinitio", "learning_rate_decay"): (
        "Exponential learning-rate decay factor applied after each ab initio solver update; 1.0 disables decay."
    ),
    ("abinitio", "momentum"): (
        "SGD momentum used for ab initio volume updates."
    ),
    ("abinitio", "loss_ema_decay"): (
        "EMA decay used for the ab initio loss metric."
    ),
    ("abinitio", "pose_rms_ema_decay"): (
        "EMA decay used for the ab initio pose-update RMS metrics."
    ),
    ("abinitio", "target_side_length_resolution"): (
        "Target side-length-derived resolution (Angstrom) used to stop side_length growth in ab initio mode."
    ),
    ("abinitio", "target_healpix_order"): (
        "Hard upper bound for the ab initio base HEALPix order. If null, it is derived from target_side_length_resolution."
    ),
    ("abinitio", "pose_rotation_stability_factor"): (
        "Scale factor applied to the current HEALPix angular step when testing rotational pose stability."
    ),
    ("abinitio", "pose_translation_stability_factor"): (
        "Scale factor applied to the current translation-grid spacing when testing translational pose stability."
    ),
    ("scheduler", "schedule_check_interval_iters"): (
        "Requested number of iterations between ab initio scheduler checkpoints; "
        "the effective interval is capped at half of the current epoch total iterations."
    ),
}


_FIELD_DEFAULT_OVERRIDE: dict[tuple[str, str], str] = {
    (
        "data",
        "default_optic_params",
    ): "{\"voltage_kv\": null, \"spherical_aberration_mm\": null, \"ctf_bfactor\": null, "
    "\"ctf_scale\": null, \"amplitude_contrast\": null, \"phase_shift_deg\": null}",
    (
        "data",
        "default_particle_params",
    ): "{\"defocus_u_angstrom\": null, \"defocus_v_angstrom\": null, \"defocus_angle_deg\": null}",
}


def _field_default_repr(section_name: str, field_name: str, dc_field) -> str:
    override = _FIELD_DEFAULT_OVERRIDE.get((section_name, field_name))
    if override is not None:
        return override

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


def _build_field_help(section_name: str, field_name: str, *, default_repr: str) -> str:
    base = _FIELD_HELP.get((section_name, field_name), f"{section_name}.{field_name}")
    return f"{base} [built-in default: {default_repr}]"


def _add_config_overrides(
    parser: argparse.ArgumentParser,
    *,
    excluded_fields: set[tuple[str, str]] | None = None,
) -> None:
    main_hints = get_type_hints(MainConfig)
    excluded_fields = excluded_fields or set()
    custom_bool_flags: dict[tuple[str, str], tuple[str, str]] = {
        ("statistics", "use_noise"): ("--use-noise", "--no-noise"),
        ("statistics", "use_prior"): ("--use-prior", "--no-prior"),
        ("scheduler", "use_cache"): ("--use-cache", "--no-cache"),
    }

    occurrences: dict[str, list[str]] = {}
    for section_field in fields(MainConfig):
        section_name = section_field.name
        section_cls = main_hints.get(section_name)
        if section_cls is None:
            continue
        for f in fields(section_cls):
            if (section_name, f.name) in excluded_fields:
                continue
            occurrences.setdefault(f.name, []).append(section_name)

    duplicated_fields = {name for name, secs in occurrences.items() if len(secs) > 1}

    alias_by_dest: dict[str, list[str]] = {
        "star_path": ["-i"],
        "output_path": ["-o"],
        "num_epochs": ["-n"],
        "batch_size": ["-b"],
        "num_workers": ["-w"],
        "level": ["--log-level"],
    }

    for section_field in fields(MainConfig):
        section_name = section_field.name
        section_cls = main_hints.get(section_name)
        if section_cls is None:
            continue

        group = parser.add_argument_group(section_name)
        section_hints = get_type_hints(section_cls)

        for f in fields(section_cls):
            if (section_name, f.name) in excluded_fields:
                continue
            if section_name == "data" and f.name == "particle_mask":
                continue

            ann = section_hints.get(f.name, Any)
            _, inner = _is_optional(ann)
            option_section: str | None = None

            if f.name in duplicated_fields:
                dest = f"{section_name}_{f.name}"
                option = f"--{section_name.replace('_', '-')}-{f.name.replace('_', '-')}"
                visible_option_strings = [option]
                hidden_option_strings: list[str] = []
            else:
                dest = f.name
                option = f"--{f.name.replace('_', '-')}"
                option_section = f"--{section_name.replace('_', '-')}-{f.name.replace('_', '-')}"
                visible_option_strings = [option, *alias_by_dest.get(dest, [])]
                hidden_option_strings = [option_section]

            if section_name == "scheduler" and f.name == "auto_local_healpix_order":
                hidden_option_strings.extend(
                    ["--auto-local-order", "--scheduler-auto-local-order"]
                )

            default_repr = _field_default_repr(section_name, f.name, f)
            kwargs: dict[str, Any] = {
                "dest": dest,
                "default": argparse.SUPPRESS,
                "help": _build_field_help(section_name, f.name, default_repr=default_repr),
            }

            custom_bool_pair = custom_bool_flags.get((section_name, f.name))
            if custom_bool_pair is not None:
                positive_option, negative_option = custom_bool_pair
                visible_option_strings = [positive_option, negative_option]
                hidden_option_strings.extend(
                    [
                        f"--no-{f.name.replace('_', '-')}",
                        f"--no-{section_name.replace('_', '-')}-{f.name.replace('_', '-')}",
                    ]
                )
                kwargs["action"] = _PairedBooleanAction
            elif inner is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
            elif inner in (int, float, str):
                kwargs["type"] = inner
            elif _looks_like_dict(inner):
                kwargs["type"] = str
            else:
                kwargs["type"] = str

            group.add_argument(*visible_option_strings, **kwargs)

            if hidden_option_strings:
                hidden_kwargs = dict(kwargs)
                hidden_kwargs["help"] = argparse.SUPPRESS
                group.add_argument(*hidden_option_strings, **hidden_kwargs)

    particle_mask_group = parser.add_argument_group("particle mask")
    particle_mask_group.add_argument(
        "--use-particle-mask",
        "--no-particle-mask",
        dest="particle_mask_enabled",
        default=argparse.SUPPRESS,
        action=_PairedBooleanAction,
        help=(
            "Enable or disable real-space particle masking during pose search. "
            "When disabled, the image is used unchanged "
            "[built-in default: False]."
        ),
    )
    particle_mask_group.add_argument(
        "--zero-particle-mask",
        "--no-zero-particle-mask",
        dest="particle_mask_zero_mask",
        default=argparse.SUPPRESS,
        action=_PairedBooleanAction,
        help=(
            "When particle masking is enabled, use `--zero-particle-mask` to fill "
            "masked-out regions with zeros, or `--no-zero-particle-mask` to sample "
            "background noise outside the particle "
            "[built-in default: True]."
        ),
    )
    particle_mask_group.add_argument(
        "--particle-mask-soft-edge-pixels",
        dest="particle_mask_soft_edge_pixels",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Soft-edge width in pixels for the particle-mask boundary used during "
            "pose search "
            "[built-in default: 5.0]."
        ),
    )
    particle_mask_group.add_argument(
        "--particle-mask-protection-disable-epochs",
        dest="particle_mask_protection_disable_epochs",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "Disable particle masking for the first N refinement epochs "
            "[built-in default: 5]."
        ),
    )
    particle_mask_group.add_argument(
        "--particle-mask-protection-radius-factor",
        dest="particle_mask_protection_radius_factor",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Multiply translation-update RMS (pixels) by this factor to get an "
            "extra mask radius in pixels; the scheduler doubles that radius, "
            "converts it to Angstrom, and adds it to data.particle_diameter "
            f"once particle masking is enabled [built-in default: "
            f"{particle_mask_utils.DEFAULT_PARTICLE_MASK_PROTECTION_RADIUS_FACTOR:.6f}, "
            f"from the `{particle_mask_utils.DEFAULT_PARTICLE_MASK_PROTECTION_COVERAGE}` "
            "coverage preset]."
        ),
    )
    particle_mask_group.add_argument(
        "--particle-mask-protection-coverage",
        dest="particle_mask_protection_coverage",
        choices=tuple(particle_mask_utils.PARTICLE_MASK_PROTECTION_COVERAGE_PRESETS),
        default=argparse.SUPPRESS,
        help=(
            "Choose a recommended protection-radius preset instead of entering a "
            "factor manually. These presets assume isotropic 2D Gaussian "
            "translation errors, for which radial coverage satisfies "
            "`coverage = 1 - exp(-factor^2)`. Presets: "
            f"{_PARTICLE_MASK_PROTECTION_PRESET_HELP}. "
            "`--particle-mask-protection-radius-factor` overrides this preset if "
            "both are provided."
        ),
    )

    ctf_group = parser.add_argument_group("ctf fallback overrides")

    ctf_group.add_argument(
        "--voltage-kv",
        dest="default_optic_params_voltage_kv",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnVoltage (kV). Overrides data.default_optic_params['voltage_kv'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--kV",
        dest="default_optic_params_voltage_kv",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    ctf_group.add_argument(
        "--spherical-aberration-mm",
        dest="default_optic_params_spherical_aberration_mm",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnSphericalAberration (mm). Overrides data.default_optic_params['spherical_aberration_mm'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--Cs",
        dest="default_optic_params_spherical_aberration_mm",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    ctf_group.add_argument(
        "--ctf-bfactor",
        dest="default_optic_params_ctf_bfactor",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnCtfBfactor. Overrides data.default_optic_params['ctf_bfactor'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--ctf-scale",
        dest="default_optic_params_ctf_scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnCtfScalefactor. Overrides data.default_optic_params['ctf_scale'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--scale",
        dest="default_optic_params_ctf_scale",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    ctf_group.add_argument(
        "--amplitude-contrast",
        dest="default_optic_params_amplitude_contrast",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnAmplitudeContrast. Overrides data.default_optic_params['amplitude_contrast'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--Q0",
        dest="default_optic_params_amplitude_contrast",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    ctf_group.add_argument(
        "--phase-shift-deg",
        dest="default_optic_params_phase_shift_deg",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnPhaseShift (deg). Overrides data.default_optic_params['phase_shift_deg'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--phase-shift",
        dest="default_optic_params_phase_shift_deg",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    ctf_group.add_argument(
        "--defocus-u-angstrom",
        dest="default_particle_params_defocus_u_angstrom",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnDefocusU (Angstrom). Overrides data.default_particle_params['defocus_u_angstrom'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--defocus-v-angstrom",
        dest="default_particle_params_defocus_v_angstrom",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnDefocusV (Angstrom). Overrides data.default_particle_params['defocus_v_angstrom'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--defocus-angle-deg",
        dest="default_particle_params_defocus_angle_deg",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnDefocusAngle (deg). Overrides data.default_particle_params['defocus_angle_deg'] [built-in default: None].",
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
    _add_config_overrides(
        homorefine,
        excluded_fields=COMMAND_HIDDEN_FIELDS["homorefine"],
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
    _add_config_overrides(
        abinitio,
        excluded_fields=COMMAND_HIDDEN_FIELDS["abinitio"],
    )

    return parser


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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "homorefine":
        run_homorefine(args)
    elif args.command == "abinitio":
        run_abinitio(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()