# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RELION External Reconstruct Wrapper for CryoFM2
"""

import os
import sys
import re
import os.path as osp
import time
import shutil
import logging
import subprocess
from filelock import FileLock, Timeout

import accelerate
from accelerate.logging import get_logger
from accelerate import PartialState

from cryofm.core.training.accelerate_utils import setup_logging
# Import MainProcessFilter first from sampling_helper to avoid circular import
from cryofm.projects.cryofm2.sampling_helper import InputData, MainProcessFilter
from cryofm.projects.cryofm2.uncond_sampling import (
    load_model, parse_args, refine3d,
    rln_ext_starfile_load_helper, detect_cryoem_file_type
)
from cryofm.projects.cryofm2.utils.infer_relion_utils import load_star, get_reconstructions


def find_corresponding_half_starfile(starfile_path):
    """
    Find the corresponding half starfile (half1 <-> half2).
    
    Args:
        starfile_path: Path to one half starfile (e.g., ...half1...star or ...half2...star)
    
    Returns:
        Path to the corresponding half starfile, or None if not found
    """
    starfile_path = osp.abspath(starfile_path)
    dirname = osp.dirname(starfile_path)
    basename = osp.basename(starfile_path)
    
    # Replace half1 with half2 or vice versa
    if "half1" in basename:
        corresponding_basename = basename.replace("half1", "half2")
    elif "half2" in basename:
        corresponding_basename = basename.replace("half2", "half1")
    else:
        return None
    
    corresponding_path = osp.join(dirname, corresponding_basename)
    
    if osp.exists(corresponding_path):
        return corresponding_path
    else:
        return None


def is_half1(starfile_path):
    """Check if the starfile is for half1."""
    return "half1" in osp.basename(starfile_path).lower()


def is_half2(starfile_path):
    """Check if the starfile is for half2."""
    return "half2" in osp.basename(starfile_path).lower()


def prepare_relion_input(starfile_path1, starfile_path2):
    """
    Prepare InputData objects from RELION starfiles.
    Uses output_path from starfile instead of constructing it.
    
    Args:
        starfile_path1: Path to first half starfile
        starfile_path2: Path to second half starfile
    
    Returns:
        input_data1, input_data2: InputData objects
    """
    from accelerate.logging import get_logger
    logger = get_logger(__name__)
    
    # Convert to absolute paths to ensure consistent path resolution
    starfile_path1 = osp.abspath(starfile_path1)
    starfile_path2 = osp.abspath(starfile_path2)
    
    logger.info(f"Loading starfile 1: {starfile_path1}")
    data1 = rln_ext_starfile_load_helper(starfile_path1)
    logger.info(f"Starfile 1 loaded successfully.")
    
    logger.info(f"Loading starfile 2: {starfile_path2}")
    data2 = rln_ext_starfile_load_helper(starfile_path2)
    logger.info(f"Starfile 2 loaded successfully.")
    
    logger.info("Computing reconstructions for half1...")
    data1_recons_unfil, data1_recons = get_reconstructions(data1)
    logger.info("Reconstructions for half1 computed.")
    
    logger.info("Computing reconstructions for half2...")
    data2_recons_unfil, data2_recons = get_reconstructions(data2)
    logger.info("Reconstructions for half2 computed.")

    input_data1 = InputData(
        file_type="STAR",
        mode=data1["mode"],
        pixel_size=data1["pixel_size"],
        original_size=data1["original_size"],
        particle_diameter=data1["particle_diameter"],
        output_path=data1["output_path"],  # Use output_path from starfile
        this_half_index=data1["this_half_index"],
        fsc_spectra=data1["fsc_spectra"],
        recons_unfil=data1_recons_unfil,
        recons=data1_recons,
        half_volume_a=data1_recons_unfil,
        half_volume_b=data2_recons_unfil,
    )
    input_data2 = InputData(
        file_type="STAR",
        mode=data2["mode"],
        pixel_size=data2["pixel_size"],
        original_size=data2["original_size"],
        particle_diameter=data2["particle_diameter"],
        output_path=data2["output_path"],  # Use output_path from starfile
        this_half_index=data2["this_half_index"],
        fsc_spectra=data2["fsc_spectra"],
        recons_unfil=data2_recons_unfil,
        recons=data2_recons,
        half_volume_a=data2_recons_unfil,
        half_volume_b=data1_recons_unfil,
    )
    
    return input_data1, input_data2


def parse_relion_wrapper_args():
    """
    Parse arguments for RELION wrapper.
    The starfile path from RELION is assumed to be the last argument.
    Other arguments are passed through to the main script.
    
    RELION typically calls:
        python relion_wrapper.py --model-dir /path/to/model starfile.star
    
    Note: The starfile must be the last argument for this wrapper to work correctly.
    """
    original_argv = sys.argv[:]
    
    # Assume starfile is always the last argument (RELION convention)
    if len(sys.argv) < 2:
        raise ValueError("No arguments provided. RELION should pass a starfile as the last argument.")
    
    starfile = sys.argv[-1]
    if not starfile.endswith('.star') or starfile.startswith('--'):
        raise ValueError(f"Last argument must be a .star file, got: {starfile}")
    
    # Extract model-dir from command line or environment
    model_dir = None
    if "--model-dir" in sys.argv:
        idx = sys.argv.index("--model-dir")
        if idx + 1 < len(sys.argv):
            candidate = sys.argv[idx + 1]
            # Make sure the value is not the starfile (which is always last)
            if candidate != starfile:
                model_dir = candidate
    
    if model_dir is None:
        model_dir = os.environ.get("CRYOFM_MODEL_DIR")
    
    if model_dir is None:
        raise ValueError("--model-dir must be specified or set via CRYOFM_MODEL_DIR environment variable")
    
    # Remove starfile from argv to prepare for parse_args()
    # This allows us to reuse the original parse_args() function
    try:
        # Create new argv without starfile (last argument)
        new_argv = sys.argv[:-1]  # All args except the last one (starfile)
        
        # Ensure model-dir is in the args (it might already be there)
        if "--model-dir" not in new_argv:
            new_argv.extend(["--model-dir", model_dir])
        
        sys.argv = new_argv
        
        # Now call the original parse_args() which will handle all other arguments
        args = parse_args()
        
        # Add starfile and set RELION-specific overrides
        args.starfile = starfile
        
        # Set output_dir from starfile if not provided
        # Also extract output_path for data-starfile-path auto-detection
        star_data = load_star(starfile)
        output_path = star_data['external_reconstruct_general']['rlnExtReconsResult']
        
        # Auto-detect --data-starfile-path from output_path if not provided
        # This is used for computing Fourier mask in inpaint mode
        # The logic: find run_it\d+ in output_path, subtract 1, and construct _data.star path
        # Example: run_it021_... -> run_it020_data.star
        # Note: when processing it001, we need run_it000_data.star (it-1)
        # This matches the behavior described in the original code comment
        if args.data_starfile_path is None:
            data_starfile_path = re.sub(
                r'^(.*?/run_it)(\d+)_.*\.mrc$',
                lambda m: f"{m.group(1)}{int(m.group(2)) - 1:0{len(m.group(2))}d}_data.star",
                output_path
            )
            # Only set if the pattern matched (i.e., output_path contains run_it\d+ pattern)
            # We don't check file existence as it might be created later by RELION
            if data_starfile_path != output_path:  # Pattern matched
                args.data_starfile_path = data_starfile_path
        
        # Set input paths (will be determined based on half1/half2)
        args.input_path = None
        args.input_path1 = None
        args.input_path2 = None
        
        return args
    finally:
        # Restore original argv
        sys.argv = original_argv


def main():
    """Main entry point for RELION wrapper."""
    args = parse_relion_wrapper_args()
    
    starfile_path = args.starfile
    
    # Check if it's a starfile
    if detect_cryoem_file_type(starfile_path) != "STAR":
        raise ValueError(f"Input file must be a STAR file, got: {starfile_path}")
    
    # CRITICAL: Handle port conflict BEFORE initializing accelerate
    # When RELION launches both half1 and half2 simultaneously with accelerate,
    # they both try to use the same default port (29500), causing EADDRINUSE error.
    # Solution: Use CRYOFM_HALF1_PORT and CRYOFM_HALF2_PORT environment variables
    # to specify different ports for each half. Defaults: half1=29500, half2=29501.
    configured_port = None
    port_source = None
    if os.environ.get("LOCAL_RANK") is not None or os.environ.get("RANK") is not None:
        # We're running under accelerate launch - set port based on half1/half2
        # This must be done BEFORE any accelerate initialization (PartialState/Accelerator)
        if is_half1(starfile_path):
            configured_port = os.environ.get("CRYOFM_HALF1_PORT", "29500")
            port_source = "CRYOFM_HALF1_PORT" if os.environ.get("CRYOFM_HALF1_PORT") else "default (29500)"
        elif is_half2(starfile_path):
            configured_port = os.environ.get("CRYOFM_HALF2_PORT", "29501")
            port_source = "CRYOFM_HALF2_PORT" if os.environ.get("CRYOFM_HALF2_PORT") else "default (29501)"
        
        if configured_port:
            # Set port (from env var or default)
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = configured_port
            if "ACCELERATE_MASTER_PORT" not in os.environ:
                os.environ["ACCELERATE_MASTER_PORT"] = configured_port
    
    # Initialize accelerate state before using logger
    # Use PartialState for lightweight initialization (sufficient for logging)
    partial_state = PartialState()
    
    # Setup logging
    setup_logging(args.log_file_path)
    logger = get_logger(__name__)
    
    # Add main process filter to all logger handlers
    # Use PartialState to determine if it's the main process (supports early logging and half2 branch)
    # Note: get_logger() returns MultiProcessAdapter which doesn't have handlers attribute,
    # so we only add filter to root handlers (setup_logging already adds handlers to root logger)
    main_process_filter = MainProcessFilter(partial_state)
    for handler in logging.root.handlers:
        handler.addFilter(main_process_filter)
    
    # Log port configuration
    if configured_port and port_source:
        current_port = os.environ.get("MASTER_PORT", "not set")
        if is_half1(starfile_path):
            logger.info(f"Half1: Using port {current_port} (from {port_source})")
        elif is_half2(starfile_path):
            logger.info(f"Half2: Using port {current_port} (from {port_source})")
    
    # Determine if this is half1 or half2
    if is_half2(starfile_path):
        """
        Lock mechanism for half2:
        
        Purpose: Prevent duplicate processing when RELION calls both half1 and half2 simultaneously.
        
        How it works:
        1. half1 will acquire a lock file based on half2's path when it starts processing
        2. half2 polls this lock file to detect if half1 has started
        3. If half2 can't acquire the lock (Timeout), it means half1 is processing -> half2 waits
        4. half2 continues polling until the lock is released (half1 completes processing)
        5. Once half1 releases the lock, half2 exits
        
        This ensures:
        - Only half1 processes both halves (avoiding duplicate computation)
        - half2 waits for half1 to complete before exiting
        - half2 handles race condition when half2 starts first by waiting for half1 to start
        """
        lock_file = starfile_path + ".processing.lock"
        
        # Poll to check if half1 has started processing
        # half1 will acquire this lock when it starts, so if we can't get it, half1 is processing
        max_wait_for_start = 30  # Maximum wait time in seconds for half1 to start
        poll_interval = 0.5  # Check every 0.5 seconds
        waited_for_start = 0
        half1_started = False
        
        logger.info(f"Detected half2 starfile. Waiting for half1 to start processing (max {max_wait_for_start}s)...")
        
        # First, wait for half1 to start (if it hasn't started yet)
        while waited_for_start < max_wait_for_start:
            try:
                # Try to acquire lock with very short timeout
                # If we can't get it, it means half1 is already processing
                lock = FileLock(lock_file, timeout=0.1)
                lock.acquire()
                # We got the lock, half1 hasn't started yet
                lock.release()
                time.sleep(poll_interval)
                waited_for_start += poll_interval
            except Timeout:
                # Lock is held by half1, so half1 has started processing
                half1_started = True
                logger.info(f"Detected half2 starfile. Half1 has started processing both halves.")
                break
        
        if not half1_started:
            # If we've waited max_wait_for_start and still can acquire the lock, half1 hasn't started
            logger.warning(f"Detected half2 starfile. Waited {max_wait_for_start}s but half1 has not started. "
                          f"Exiting to avoid duplicate processing. "
                          f"If half1 processing failed, you may need to run half1 manually.")
            sys.exit(0)
        
        # Now wait for half1 to complete processing (lock to be released)
        logger.info(f"Detected half2 starfile. Waiting for half1 to complete processing...")
        while True:
            try:
                # Try to acquire lock with short timeout
                # If we can get it, it means half1 has finished and released the lock
                lock = FileLock(lock_file, timeout=0.1)
                lock.acquire()
                # We got the lock, half1 has completed processing
                lock.release()
                logger.info(f"Detected half2 starfile. Half1 has completed processing. Exiting.")
                sys.exit(0)
            except Timeout:
                # Lock is still held by half1, continue waiting
                time.sleep(poll_interval)
    
    elif is_half1(starfile_path):
        # For half1, find corresponding half2 and process both
        logger.info(f"Detected half1 starfile: {starfile_path}")
        
        # Find corresponding half2
        half2_path = find_corresponding_half_starfile(starfile_path)
        if half2_path is None:
            raise FileNotFoundError(
                f"Could not find corresponding half2 starfile for {starfile_path}. "
                f"Expected a file with 'half2' instead of 'half1' in the same directory."
            )
        
        logger.info(f"Found corresponding half2 starfile: {half2_path}")
        
        # Setup accelerator first to enable process coordination
        # This is critical when using accelerate launch with multiple processes
        logger.info("Setting up accelerator...")
        accelerator = accelerate.Accelerator()
        logger.info(f"Accelerator setup complete. Device: {accelerator.device}, Process: {accelerator.process_index}/{accelerator.num_processes}")
        
        # Update filter to use accelerator (more accurate, supports is_main_process)
        # Remove old filter and add new one
        # Note: get_logger() returns MultiProcessAdapter which doesn't have handlers attribute,
        # so we only update root handlers
        for handler in logging.root.handlers:
            handler.filters = [f for f in handler.filters if not isinstance(f, MainProcessFilter)]
            handler.addFilter(MainProcessFilter(accelerator))
        
        # Use file lock to prevent half2 from processing
        # The lock file is based on half2 path so half2 can detect it
        # This lock is held for the entire processing duration (24 hours timeout)
        # Only main process should acquire the lock to avoid conflicts
        lock_file = half2_path + ".processing.lock"
        lock_timeout = 24 * 3600  # 24 hours in seconds
        lock = FileLock(lock_file, timeout=lock_timeout)
        
        try:
            # Only main process acquires lock
            if accelerator.is_main_process:
                # Acquire lock at the start of processing
                logger.info(f"Acquiring lock file {lock_file} for processing (timeout: {lock_timeout}s)...")
                lock.acquire()
                logger.info(f"Lock acquired. Starting processing...")
            
            # Synchronize all processes - ensure lock is acquired before data preparation
            accelerator.wait_for_everyone()
            
            # All processes prepare input data (each process needs its own copy)
            # This is safe because file I/O operations are generally thread-safe
            # and each process will read the same files independently
            logger.info("Preparing input data from starfiles...")
            input_data1, input_data2 = prepare_relion_input(
                starfile_path, half2_path, 
            )
            logger.info("Input data prepared successfully.")
            
            # Synchronize all processes after data preparation
            accelerator.wait_for_everyone()
            
            logger.info("Loading model...")
            lit_model_, unet_, scheduler_ = load_model(args, device=accelerator.device)
            logger.info("Model loaded successfully.")
            
            # Synchronize before processing
            accelerator.wait_for_everyone()
            
            # Process both halves
            logger.info("Starting refine3d processing...")
            refine3d(
                args,
                input_data1=input_data1,
                input_data2=input_data2,
                scheduler=scheduler_,
                unet=unet_,
                acl=accelerator,
                logger=logger
            )
            
            logger.info("Waiting for all processes to complete...")
            accelerator.wait_for_everyone()
            logger.info("Processing completed successfully.")
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            raise
        finally:
            # Only main process releases the lock
            if accelerator.is_main_process:
                # Release lock after refine3d completes
                logger.info(f"Releasing lock file {lock_file}...")
                lock.release()
                logger.info(f"Lock released.")
            # Ensure all processes wait before exiting
            accelerator.wait_for_everyone()
    
    else:
        raise ValueError(
            f"Starfile path must contain 'half1' or 'half2'. Got: {starfile_path}"
        )


def launch_with_accelerate():
    """
    Launch this script with accelerate launch, automatically setting the port
    based on half1/half2 detection and using all available GPUs.
    """
    # Find the starfile in arguments to determine half1/half2
    starfile = None
    for arg in sys.argv[1:]:
        if arg.endswith('.star') and not arg.startswith('--'):
            starfile = arg
            break
    
    # Determine port based on half1/half2
    if starfile:
        if is_half1(starfile):
            port = os.environ.get("CRYOFM_HALF1_PORT", "29500")
            print(f"[relion_wrapper.py] Half1 detected: Using port {port}")
        elif is_half2(starfile):
            port = os.environ.get("CRYOFM_HALF2_PORT", "29501")
            print(f"[relion_wrapper.py] Half2 detected: Using port {port}")
        else:
            port = os.environ.get("MASTER_PORT", "29500")
            print(f"[relion_wrapper.py] Warning: Could not detect half1/half2, using port {port}")
    else:
        port = os.environ.get("MASTER_PORT", "29500")
        print(f"[relion_wrapper.py] No starfile found, using port {port}")
    
    # Set environment variables for accelerate launch (as backup)
    os.environ["MASTER_PORT"] = port
    os.environ["ACCELERATE_MASTER_PORT"] = port
    
    # Parse arguments - extract accelerate launch specific options, pass rest to script
    accelerate_args = []
    script_args = []
    skip_next = False
    user_specified_port = False
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        
        # Extract accelerate launch options
        if arg == "--main_process_port" and i < len(sys.argv) - 1:
            # User explicitly set port, use it instead of auto-detection
            accelerate_args.extend(["--main_process_port", sys.argv[i + 1]])
            skip_next = True
            user_specified_port = True
        elif arg.startswith("--main_process_port="):
            accelerate_args.append(arg)
            user_specified_port = True
        elif arg.startswith("--num_processes") or arg == "--num_processes":
            # Ignore num_processes - we'll use all GPUs automatically
            if arg == "--num_processes" and i < len(sys.argv) - 1:
                skip_next = True
            # Don't add to accelerate_args or script_args
        else:
            # All other arguments go to the script
            script_args.append(arg)
    
    # If user didn't specify port, add our auto-detected port to accelerate_args
    if not user_specified_port:
        accelerate_args.extend(["--main_process_port", port])
    
    # Build the command: accelerate launch [options] python relion_wrapper.py [script_args]
    # Use __file__ to get the absolute path to this script
    # Note: accelerate launch will automatically use all available GPUs if num_processes is not specified
    # Find accelerate command - try to locate it in the Python environment
    script_path = osp.abspath(__file__)
    
    # Try to find accelerate command
    accelerate_cmd = None
    
    # Method 1: Try shutil.which to find accelerate in PATH
    accelerate_cmd = shutil.which("accelerate")
    
    # Method 2: If not found, try in the same directory as Python executable
    if not accelerate_cmd:
        python_dir = osp.dirname(sys.executable)
        possible_accelerate = osp.join(python_dir, "accelerate")
        if osp.exists(possible_accelerate) and os.access(possible_accelerate, os.X_OK):
            accelerate_cmd = possible_accelerate
    
    # Method 3: Try common conda/virtualenv locations
    if not accelerate_cmd:
        for base_dir in [sys.prefix, sys.exec_prefix]:
            possible_accelerate = osp.join(base_dir, "bin", "accelerate")
            if osp.exists(possible_accelerate) and os.access(possible_accelerate, os.X_OK):
                accelerate_cmd = possible_accelerate
                break
    
    if not accelerate_cmd:
        raise RuntimeError(
            "Could not find 'accelerate' command. Please ensure accelerate is installed "
            "and available in PATH. You can install it with: pip install accelerate"
        )
    
    # Verify accelerate command is executable
    if not os.access(accelerate_cmd, os.X_OK):
        raise RuntimeError(
            f"Found accelerate command at {accelerate_cmd} but it is not executable. "
            f"Please check the file permissions."
        )
    
    # Build command: accelerate launch [accelerate_args] python script_path [script_args]
    # Note: accelerate launch automatically detects where script args start
    cmd = [accelerate_cmd, "launch"] + accelerate_args + [
        script_path,
    ] + script_args
    
    print(f"[relion_wrapper.py] Launching with all available GPUs")
    print(f"[relion_wrapper.py] Accelerate command: {accelerate_cmd}")
    print(f"[relion_wrapper.py] Port: {port}")
    print(f"[relion_wrapper.py] Full command: {' '.join(cmd)}")
    
    # Execute accelerate launch
    # Use shell=False to avoid shell interpretation issues
    try:
        sys.exit(subprocess.call(cmd, shell=False))
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Failed to execute accelerate command: {accelerate_cmd}\n"
            f"Error: {e}\n"
            f"Please ensure accelerate is properly installed: pip install accelerate"
        ) from e


if __name__ == "__main__":
    # Check if we're already running under accelerate launch
    # If not, launch with accelerate (using all available GPUs). If yes, run main() directly.
    if os.environ.get("LOCAL_RANK") is None and os.environ.get("RANK") is None:
        # Not in accelerate launch yet - automatically launch with accelerate
        # This will use all available GPUs automatically
        launch_with_accelerate()
    else:
        # Already in accelerate launch, run main() directly
        main()
