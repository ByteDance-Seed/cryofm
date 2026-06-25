"""
CryoSeed external reconstruct wrapper for CryoFM2.

This adapts the STAR files exported by `cryoseed homorefine --external-reconstruct`
to the CryoFM2 inference CLI. HomoRefine launches one wrapper per half (`half0` and
`half1`); this wrapper lets the `half0` invocation process both halves while `half1`
waits and exits.

Example:
export CRYOSEED_EXTERNAL_RECONSTRUCT_EXECUTABLE="python /path/to/cryoseed_wrapper.py --model-dir /path/to/model_dir --op denoise"
"""

import argparse
import logging
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock, Timeout
from huggingface_hub import snapshot_download

from cryofm.projects.cryofm2.utils.infer_relion_utils import load_star


def _detect_half(starfile_path: str) -> str | None:
    basename_lower = osp.basename(starfile_path).lower()
    if "half0" in basename_lower:
        return "half0"
    if "half1" in basename_lower:
        return "half1"
    return None


def find_corresponding_half_starfile(starfile_path: str) -> str | None:
    starfile_path = osp.abspath(starfile_path)
    dirname = osp.dirname(starfile_path)
    basename = osp.basename(starfile_path)
    half_name = _detect_half(starfile_path)
    if half_name is None:
        return None

    other_half_name = "half1" if half_name == "half0" else "half0"
    start_idx = basename.lower().index(half_name)
    corresponding_basename = (
        basename[:start_idx] + other_half_name + basename[start_idx + len(half_name):]
    )
    corresponding_path = osp.join(dirname, corresponding_basename)
    if osp.exists(corresponding_path):
        return corresponding_path
    return None


def _resolve_starfile_path(work_dir: str, star_file: str) -> str:
    if osp.isabs(star_file):
        return osp.abspath(star_file)
    return osp.abspath(osp.join(work_dir, star_file))


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_dir(args: argparse.Namespace) -> str:
    model_dir = args.model_dir or os.environ.get("CRYOFM_MODEL_DIR")
    model_id = args.model_id or os.environ.get("CRYOFM_MODEL_ID")

    if model_dir and model_id:
        raise ValueError("Please specify only one of --model-dir or --model-id.")
    if model_dir:
        resolved_dir = osp.abspath(model_dir)
    else:
        if not model_id:
            raise ValueError(
                "Either --model-dir/CRYOFM_MODEL_DIR or --model-id/CRYOFM_MODEL_ID must be specified."
            )
        model_revision = args.model_revision or os.environ.get("CRYOFM_MODEL_REVISION")
        hf_cache_dir = args.hf_cache_dir or os.environ.get("CRYOFM_HF_CACHE_DIR")
        local_files_only = args.local_files_only or _is_truthy_env(
            os.environ.get("CRYOFM_HF_LOCAL_FILES_ONLY")
        )
        resolved_dir = snapshot_download(
            repo_id=model_id,
            revision=model_revision,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
            allow_patterns=["config.yaml", "model.safetensors"],
        )

    config_path = osp.join(resolved_dir, "config.yaml")
    weights_path = osp.join(resolved_dir, "model.safetensors")
    if not osp.isfile(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    if not osp.isfile(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    return resolved_dir


def _resolve_op(args: argparse.Namespace) -> str:
    op = args.op or os.environ.get("CRYOFM_OP") or "denoise"
    return op


def _build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--star-file", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--op")
    return parser


def parse_cryoseed_wrapper_args() -> dict:
    args, passthrough_args = _build_wrapper_parser().parse_known_args(sys.argv[1:])
    model_dir = _resolve_model_dir(args)
    op = _resolve_op(args)

    work_dir = osp.abspath(args.work_dir)
    starfile = _resolve_starfile_path(work_dir, args.star_file)
    if not osp.isfile(starfile):
        raise FileNotFoundError(f"STAR file does not exist: {starfile}")

    return {
        "work_dir": work_dir,
        "model_dir": model_dir,
        "op": op,
        "starfile": starfile,
        "passthrough_args": passthrough_args,
    }


def _resolve_result_path(starfile_path: str, work_dir: str, result_path: str) -> str:
    if osp.isabs(result_path):
        return osp.abspath(result_path)

    candidates = [
        osp.abspath(osp.join(osp.dirname(starfile_path), result_path)),
        osp.abspath(osp.join(work_dir, result_path)),
    ]
    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate
    return candidates[0]


def _ensure_parent_dir(path: str) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _get_generated_output_path(work_dir: str, input_map_path: str) -> str:
    stem = Path(input_map_path).stem
    if stem.endswith("_external_reconstruct"):
        output_stem = stem
    else:
        output_stem = f"{stem}_external_reconstruct"
    return osp.join(work_dir, f"{output_stem}.mrc")


def _is_lock_available(lock_file: str) -> bool:
    try:
        lock = FileLock(lock_file, timeout=0.1)
        lock.acquire()
        lock.release()
        return True
    except Timeout:
        return False


def _split_accelerate_args(argv: list[str], default_port: str) -> tuple[list[str], list[str]]:
    accelerate_args = []
    script_args = []
    skip_next = False
    user_specified_port = False

    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if arg == "--main_process_port" and idx + 1 < len(argv):
            accelerate_args.extend(["--main_process_port", argv[idx + 1]])
            skip_next = True
            user_specified_port = True
        elif arg.startswith("--main_process_port="):
            accelerate_args.append(arg)
            user_specified_port = True
        elif arg == "--num_processes":
            if idx + 1 < len(argv):
                skip_next = True
        elif arg.startswith("--num_processes="):
            continue
        else:
            script_args.append(arg)

    if not user_specified_port:
        accelerate_args.extend(["--main_process_port", default_port])

    return accelerate_args, script_args


def _filter_conflicting_sampling_args(argv: list[str]) -> list[str]:
    filtered = []
    skip_next = False
    flags_with_values = {
        "-i",
        "--input-path",
        "-i1",
        "--input-path1",
        "-i2",
        "--input-path2",
        "-o",
        "--output-dir",
        "--model-dir",
        "--op",
    }
    flags_without_values = {
        "--norm-grad",
        "--use-lamb-w",
    }

    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if arg in flags_with_values:
            if idx + 1 < len(argv):
                skip_next = True
            continue
        if any(arg.startswith(flag + "=") for flag in flags_with_values):
            continue
        if arg in flags_without_values:
            continue
        filtered.append(arg)

    return filtered


def _find_accelerate_command() -> str:
    accelerate_cmd = shutil.which("accelerate")
    if not accelerate_cmd:
        python_dir = osp.dirname(sys.executable)
        candidate = osp.join(python_dir, "accelerate")
        if osp.exists(candidate) and os.access(candidate, os.X_OK):
            accelerate_cmd = candidate

    if not accelerate_cmd:
        for base_dir in [sys.prefix, sys.exec_prefix]:
            candidate = osp.join(base_dir, "bin", "accelerate")
            if osp.exists(candidate) and os.access(candidate, os.X_OK):
                accelerate_cmd = candidate
                break

    if not accelerate_cmd:
        raise RuntimeError(
            "Could not find 'accelerate' command. Please ensure accelerate is installed "
            "and available in PATH."
        )
    if not os.access(accelerate_cmd, os.X_OK):
        raise RuntimeError(
            f"Found accelerate command at {accelerate_cmd} but it is not executable."
        )
    return accelerate_cmd


def _replace_output(source_path: str, target_path: str) -> None:
    source_path = osp.abspath(source_path)
    target_path = osp.abspath(target_path)
    _ensure_parent_dir(target_path)

    if osp.samefile(source_path, target_path) if osp.exists(source_path) and osp.exists(target_path) else False:
        return

    if osp.lexists(target_path):
        if osp.isdir(target_path) and not osp.islink(target_path):
            raise IsADirectoryError(f"Cannot replace directory with file: {target_path}")
        os.remove(target_path)

    os.replace(source_path, target_path)


def prepare_half_maps(
    starfile_path0: str,
    starfile_path1: str,
    work_dir: str,
) -> dict:
    logger = logging.getLogger(__name__)

    starfile_path0 = osp.abspath(starfile_path0)
    starfile_path1 = osp.abspath(starfile_path1)
    work_dir = osp.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    star0 = load_star(starfile_path0)
    star1 = load_star(starfile_path1)
    result_path0 = _resolve_result_path(
        starfile_path0,
        work_dir,
        star0["external_reconstruct_general"]["rlnExtReconsResult"],
    )
    result_path1 = _resolve_result_path(
        starfile_path1,
        work_dir,
        star1["external_reconstruct_general"]["rlnExtReconsResult"],
    )

    if not osp.isfile(result_path0):
        raise FileNotFoundError(f"Half0 reconstruction file does not exist: {result_path0}")
    if not osp.isfile(result_path1):
        raise FileNotFoundError(f"Half1 reconstruction file does not exist: {result_path1}")

    logger.info("Using half0 reconstruction from starfile: %s", result_path0)
    logger.info("Using half1 reconstruction from starfile: %s", result_path1)

    return {
        "half_map_1": result_path0,
        "half_map_2": result_path1,
        "result_path0": result_path0,
        "result_path1": result_path1,
    }


def launch_uncond_sampling(
    work_dir: str,
    model_dir: str,
    op: str,
    half_map_1: str,
    half_map_2: str,
    passthrough_args: list[str],
    port: str,
) -> None:
    logger = logging.getLogger(__name__)
    accelerate_cmd = _find_accelerate_command()
    accelerate_args, script_args = _split_accelerate_args(passthrough_args, port)
    script_args = _filter_conflicting_sampling_args(script_args)

    cmd = [
        accelerate_cmd,
        "launch",
        *accelerate_args,
        "-m",
        "cryofm.projects.cryofm2.uncond_sampling",
        "-i1",
        half_map_1,
        "-i2",
        half_map_2,
        "-o",
        work_dir,
        "--model-dir",
        model_dir,
        "--op",
        op,
        "--norm-grad",
        "--use-lamb-w",
        *script_args,
    ]

    env = os.environ.copy()
    env["MASTER_PORT"] = env.get("MASTER_PORT", port)
    env["ACCELERATE_MASTER_PORT"] = env.get("ACCELERATE_MASTER_PORT", port)

    logger.info("Launching CryoFM2 via accelerate")
    logger.info("Accelerate command: %s", accelerate_cmd)
    logger.info("Port: %s", env["MASTER_PORT"])
    logger.info("Full command: %s", " ".join(cmd))

    try:
        subprocess.check_call(cmd, shell=False, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to execute accelerate command: {accelerate_cmd}\nError: {exc}"
        ) from exc


def publish_results(
    work_dir: str,
    half_map_1: str,
    half_map_2: str,
    result_path0: str,
    result_path1: str,
) -> None:
    generated_half0 = _get_generated_output_path(work_dir, half_map_1)
    generated_half1 = _get_generated_output_path(work_dir, half_map_2)

    if not osp.isfile(generated_half0):
        raise FileNotFoundError(f"Expected generated file not found: {generated_half0}")
    if not osp.isfile(generated_half1):
        raise FileNotFoundError(f"Expected generated file not found: {generated_half1}")

    _replace_output(generated_half0, result_path0)
    _replace_output(generated_half1, result_path1)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[cryoseed_wrapper.py] %(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parse_cryoseed_wrapper_args()
    starfile_path = args["starfile"]
    half_name = _detect_half(starfile_path)

    if not starfile_path.lower().endswith(".star"):
        raise ValueError(f"Input file must be a STAR file, got: {starfile_path}")

    if half_name == "half1":
        lock_file = starfile_path + ".processing.lock"
        max_wait_for_start = 30
        poll_interval = 0.5

        logger.info(
            "Detected half1 starfile. Waiting for half0 to start processing (max %ss)...",
            max_wait_for_start,
        )

        start_time = time.monotonic()
        while time.monotonic() - start_time < max_wait_for_start:
            if not _is_lock_available(lock_file):
                logger.info("Detected half1 starfile. Half0 has started processing both halves.")
                break
            time.sleep(poll_interval)
        else:
            logger.warning(
                "Detected half1 starfile. Waited %ss but half0 has not started. "
                "Exiting to avoid duplicate processing.",
                max_wait_for_start,
            )
            sys.exit(0)

        logger.info("Detected half1 starfile. Waiting for half0 to complete processing...")
        while not _is_lock_available(lock_file):
            time.sleep(poll_interval)

        logger.info("Detected half1 starfile. Half0 has completed processing. Exiting.")
        sys.exit(0)

    if half_name != "half0":
        raise ValueError(
            f"STAR file path must contain 'half0' or 'half1'. Got: {starfile_path}"
        )

    port = os.environ.get("CRYOFM_HALF0_PORT", "19500")
    logger.info("Detected half0 starfile: %s", starfile_path)
    half1_path = find_corresponding_half_starfile(starfile_path)
    if half1_path is None:
        raise FileNotFoundError(
            f"Could not find corresponding half1 starfile for {starfile_path}. "
            "Expected a file with 'half1' instead of 'half0' in the same directory."
        )

    logger.info("Found corresponding half1 starfile: %s", half1_path)

    lock_file = half1_path + ".processing.lock"
    lock_timeout = 24 * 3600
    lock = FileLock(lock_file, timeout=lock_timeout)

    try:
        logger.info(
            "Acquiring lock file %s for processing (timeout: %ss)...",
            lock_file,
            lock_timeout,
        )
        lock.acquire()
        logger.info("Lock acquired. Preparing half maps...")

        prepared = prepare_half_maps(starfile_path, half1_path, args["work_dir"])
        logger.info(
            "Prepared half maps: %s, %s",
            prepared["half_map_1"],
            prepared["half_map_2"],
        )

        launch_uncond_sampling(
            work_dir=args["work_dir"],
            model_dir=args["model_dir"],
            op=args["op"],
            half_map_1=prepared["half_map_1"],
            half_map_2=prepared["half_map_2"],
            passthrough_args=args["passthrough_args"],
            port=port,
        )
        logger.info("CryoFM2 CLI finished successfully. Publishing results...")

        publish_results(
            work_dir=args["work_dir"],
            half_map_1=prepared["half_map_1"],
            half_map_2=prepared["half_map_2"],
            result_path0=prepared["result_path0"],
            result_path1=prepared["result_path1"],
        )
        logger.info("Processing completed successfully.")
    except Exception as exc:
        logger.error("Error during processing: %s", exc, exc_info=True)
        raise
    finally:
        if lock.is_locked:
            lock.release()
        logger.info("Lock released.")


if __name__ == "__main__":
    main()