import argparse
import json
from dataclasses import MISSING, fields
from typing import Any, get_args, get_origin, get_type_hints

from cryoseed._version import __version__
from cryoseed.config import MainConfig


class _HelpFormatter(argparse.RawTextHelpFormatter):
    pass


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
        "'{\"kV\":300,\"Cs\":2.7,\"Bfac\":0,\"scale\":1,\"Q0\":0.1,\"phase_shift\":0}'."
    ),
    ("data", "default_particle_params"): (
        "Fallback particle-level CTF fields as a JSON object string, e.g. "
        "'{\"DeltafU\":10000,\"DeltafV\":10000,\"azimuthal_angle\":0}'."
    ),

    ("logging", "log_dir"): "Log directory (default: output_path/logs).",
    ("logging", "log_prefix"): "Log filename prefix.",

    ("reconstruction", "num_volumes"): "Number of volumes/classes K (homorefine requires K=1).",
    ("reconstruction", "external_reconstruct"): "Enable external reconstruction data export under output_path/external_reconstruct.",
    ("reconstruction", "full_backprojection"): "Use the full Fourier image radius during backprojection instead of the current side_length-limited radius.",
    ("reconstruction", "requires_grad"): "Whether the volume requires gradients.",
    ("reconstruction", "requires_accum"): "Whether to allocate/use accumulation buffers.",
    ("reconstruction", "backproject_chunk"): "Chunk size over poses in backprojection (memory/speed tradeoff).",
    ("reconstruction", "accumulate_chunk"): "Chunk size over poses in noise accumulation (memory/speed tradeoff).",

    ("statistics", "use_noise"): "Enable noise variance spectrum estimation.",
    ("statistics", "use_prior"): "Enable prior variance spectrum regularization.",
    ("statistics", "init_variance"): "Initial value for noise/prior variance spectra.",
    ("statistics", "precision_eps"): "Clamp epsilon used when forming precision = 1/variance.",
    ("statistics", "tail_floor"): "Floor value for the high-frequency exponential tail of the prior.",
    ("statistics", "init_lowpass_cutoff"): "Low-pass size L (pixels) used to initialize the prior tail.",

    ("refinement", "num_epochs"): "Number of refinement epochs.",
    ("refinement", "fsc_threshold"): "FSC threshold used to estimate resolution.",
    ("refinement", "init_lowpass_angstrom"): "Initial low-pass resolution (Angstrom) for setting side_length.",

    ("scheduler", "confidence_threshold"): "avg_confidence threshold for aggressive side_length growth.",
    ("scheduler", "increase_radius_step"): "Default radius increment in frequency marching (pixels).",
    ("scheduler", "increase_radius_aggressive_factor"): "Extra radius increment factor when confident.",
    ("scheduler", "base_healpix_order"): (
        "Starting HEALPix order used while the scheduler stays in global HEALPix search."
    ),
    ("scheduler", "auto_local_healpix_order"): (
        "Switch from global HEALPix search to local Euler search when output_healpix_order reaches this value."
    ),
    ("scheduler", "use_cache"): "Enable projection cache (memory/SSD) for pose search.",
    ("scheduler", "cache_max_healpix_order"): "Enable caching only when healpix_order <= this value.",
    ("scheduler", "ssd_cache_min_side_length"): "Use SSD cache when side_length >= this value; otherwise memory cache.",

    ("pose_search", "init_healpix_order"): "Initial HEALPix order.",
    ("pose_search", "k_steps"): "Local neighborhood radius k (controls (2k+1)^3 / (2k+1)^2 expansions).",
    ("pose_search", "t_extent"): "Translation search extent in pixels.",
    ("pose_search", "t_ngrid"): "Translation grid density parameter.",
    ("pose_search", "t_xshift"): "Translation grid x offset (pixels).",
    ("pose_search", "t_yshift"): "Translation grid y offset (pixels).",
    ("pose_search", "pose_chunk_factor"): "Chunking factor for projection/translation computation (memory/speed tradeoff).",
    ("pose_search", "max_candidates"): "Max number of pose candidates kept per image.",
    ("pose_search", "mse_chunk"): "Chunk size for MSE/likelihood evaluation.",
    ("pose_search", "candidate_select_threshold"): "Cumulative probability threshold for candidate selection.",
    ("pose_search", "renormalize_sel_prob"): "Renormalize selected per-image candidate probabilities to sum to 1 after truncation.",

    ("reproduce", "seed"): "Random seed.",
    ("reproduce", "deterministic"): "Enable deterministic mode (may reduce performance).",
}


_FIELD_DEFAULT_OVERRIDE: dict[tuple[str, str], str] = {
    (
        "data",
        "default_optic_params",
    ): "{\"kV\": null, \"Cs\": null, \"Bfac\": null, \"scale\": null, \"Q0\": null, \"phase_shift\": null}",
    (
        "data",
        "default_particle_params",
    ): "{\"DeltafU\": null, \"DeltafV\": null, \"azimuthal_angle\": null}",
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


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    main_hints = get_type_hints(MainConfig)

    occurrences: dict[str, list[str]] = {}
    for section_field in fields(MainConfig):
        section_name = section_field.name
        section_cls = main_hints.get(section_name)
        if section_cls is None:
            continue
        for f in fields(section_cls):
            occurrences.setdefault(f.name, []).append(section_name)

    duplicated_fields = {name for name, secs in occurrences.items() if len(secs) > 1}

    alias_by_dest: dict[str, list[str]] = {
        "star_path": ["-i"],
        "output_path": ["-o"],
        "num_epochs": ["-n"],
        "batch_size": ["-b"],
        "num_workers": ["-w"],
    }

    for section_field in fields(MainConfig):
        section_name = section_field.name
        section_cls = main_hints.get(section_name)
        if section_cls is None:
            continue

        group = parser.add_argument_group(section_name)
        section_hints = get_type_hints(section_cls)

        for f in fields(section_cls):
            ann = section_hints.get(f.name, Any)
            _, inner = _is_optional(ann)

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

            if section_name == "reconstruction" and f.name == "external_reconstruct":
                kwargs["action"] = "store_true"
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

    ctf_group = parser.add_argument_group("ctf fallback overrides")

    ctf_group.add_argument(
        "--kV",
        dest="default_optic_params_kV",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnVoltage (kV). Overrides data.default_optic_params['kV'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--Cs",
        dest="default_optic_params_Cs",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnSphericalAberration (mm). Overrides data.default_optic_params['Cs'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--Bfac",
        dest="default_optic_params_Bfac",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnCtfBfactor. Overrides data.default_optic_params['Bfac'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--scale",
        dest="default_optic_params_scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnCtfScalefactor. Overrides data.default_optic_params['scale'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--Q0",
        dest="default_optic_params_Q0",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnAmplitudeContrast. Overrides data.default_optic_params['Q0'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--phase-shift",
        dest="default_optic_params_phase_shift",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnPhaseShift (deg). Overrides data.default_optic_params['phase_shift'] [built-in default: None].",
    )

    ctf_group.add_argument(
        "--DeltafU",
        dest="default_particle_params_DeltafU",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnDefocusU (Angstrom). Overrides data.default_particle_params['DeltafU'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--DeltafV",
        dest="default_particle_params_DeltafV",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnDefocusV (Angstrom). Overrides data.default_particle_params['DeltafV'] [built-in default: None].",
    )
    ctf_group.add_argument(
        "--defocus-angle",
        dest="default_particle_params_azimuthal_angle",
        type=float,
        default=argparse.SUPPRESS,
        help="Fallback for missing rlnDefocusAngle (deg). Overrides data.default_particle_params['azimuthal_angle'] [built-in default: None].",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryoseed",
        description="Cryo-EM reconstruction toolkit",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    homorefine = subparsers.add_parser(
        "homorefine",
        help="Run homogeneous refinement",
        description=(
            "Run homogeneous refinement.\n\n"
            "Precedence: CLI options > config file > built-in defaults.\n"
            "The config file is optional.\n"
            "Options not provided on the command line fall back to the config file,\n"
            "and then to built-in defaults."
        ),
        formatter_class=_HelpFormatter,
    )
    homorefine.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=argparse.SUPPRESS,
        help="Path to a YAML/JSON config file. Command-line arguments override config values.",
    )
    homorefine.add_argument(
        "--data-parallel-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Data-parallel group size (dp). Must satisfy dp * cp == WORLD_SIZE.",
    )
    homorefine.add_argument(
        "--compute-parallel-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Compute-parallel group size (cp). Must satisfy dp * cp == WORLD_SIZE.",
    )
    resume_group = homorefine.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        type=str,
        default=argparse.SUPPRESS,
        help="Resume homorefine from a checkpoint and continue from its next epoch.",
    )
    resume_group.add_argument(
        "--auto-resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Resume from output_path/checkpoints/latest.pt if it exists; otherwise start fresh.",
    )
    _add_config_overrides(homorefine)

    return parser


def run_homorefine(args) -> None:
    config = MainConfig.from_cli_args(args)

    runtime = setup_runtime(
        data_parallel_size=getattr(args, "data_parallel_size", None),
        compute_parallel_size=getattr(args, "compute_parallel_size", None),
    )
    try:
        if runtime.rank == 0:
            saved_config_path, snapshot_config_path = config.save_output_config()
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

    if args.command == "homorefine":
        run_homorefine(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()