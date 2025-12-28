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
CryoFM CLI entry point
"""

import os
import sys
import subprocess


def _is_accelerate_launched():
    """Check if we're running under accelerate launch"""
    return os.environ.get("ACCELERATE_USE_CPU") is not None or \
           os.environ.get("LOCAL_RANK") is not None or \
           os.environ.get("RANK") is not None


def _run_with_accelerate(module_path, remaining_args, num_processes=None, main_process_port=None):
    """Run command with accelerate launch"""
    accelerate_cmd = ["accelerate", "launch"]
    
    if num_processes:
        accelerate_cmd.extend(["--num_processes", str(num_processes)])
    if main_process_port:
        accelerate_cmd.extend(["--main_process_port", str(main_process_port)])
    
    cmd = accelerate_cmd + [sys.executable, "-m", module_path] + remaining_args
    
    # Run with subprocess
    sys.exit(subprocess.call(cmd))


def _run_module_directly(module_path, remaining_args):
    """Run module directly (single GPU or already under accelerate)"""
    cmd = [sys.executable, "-m", module_path] + remaining_args
    sys.exit(subprocess.call(cmd))


def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CryoFM Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Denoise with single GPU
  cfm denoise -i1 half1.mrc -i2 half2.mrc -o ./output --model-dir ./model

  # Denoise with multiple GPUs
  cfm denoise --num_processes 4 -i1 half1.mrc -i2 half2.mrc -o ./output --model-dir ./model

  # Enhance with single GPU
  cfm enhance -i input.mrc -o ./output --model-dir ./model --output-tag 1

  # Enhance with multiple GPUs
  cfm enhance --num_processes 2 -i input.mrc -o ./output --model-dir ./model --output-tag 1

  # Use accelerate launch directly (for more control)
  accelerate launch --num_processes 4 cfm denoise -i1 half1.mrc -i2 half2.mrc -o ./output --model-dir ./model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)
    
    # Denoise subcommand
    denoise_parser = subparsers.add_parser(
        'denoise',
        help='Apply denoising to cryo-EM density maps',
        description='Denoise cryo-EM density maps using CryoFM2 unconditional model. '
                     'This command wraps python -m cryofm.projects.cryofm2.uncond_sampling'
    )
    denoise_parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of GPUs to use (will use accelerate launch if specified)"
    )
    denoise_parser.add_argument(
        "--main_process_port",
        type=int,
        default=None,
        help="Main process port for accelerate launch"
    )
    
    # Enhance subcommand
    enhance_parser = subparsers.add_parser(
        'enhance',
        help='Apply style enhancement to cryo-EM density maps',
        description='Enhance cryo-EM density maps using CryoFM2 conditional model. '
                   'This command wraps python -m cryofm.projects.cryofm2.cond_sampling'
    )
    enhance_parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of GPUs to use (will use accelerate launch if specified)"
    )
    enhance_parser.add_argument(
        "--main_process_port",
        type=int,
        default=None,
        help="Main process port for accelerate launch"
    )
    
    # First parse to get the command
    args, remaining_argv = parser.parse_known_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'denoise':
        # Parse again with the denoise parser to get our specific args
        # remaining_argv already contains all arguments after 'denoise'
        denoise_args, remaining_args = denoise_parser.parse_known_args(remaining_argv)
        num_processes = denoise_args.num_processes
        main_process_port = denoise_args.main_process_port
        module_path = "cryofm.projects.cryofm2.uncond_sampling"
        
        if num_processes is not None or main_process_port is not None:
            _run_with_accelerate(module_path, remaining_args, num_processes, main_process_port)
        else:
            _run_module_directly(module_path, remaining_args)
                
    elif args.command == 'enhance':
        # Parse again with the enhance parser to get our specific args
        # remaining_argv already contains all arguments after 'enhance'
        enhance_args, remaining_args = enhance_parser.parse_known_args(remaining_argv)
        num_processes = enhance_args.num_processes
        main_process_port = enhance_args.main_process_port
        module_path = "cryofm.projects.cryofm2.cond_sampling"
        
        if num_processes is not None or main_process_port is not None:
            _run_with_accelerate(module_path, remaining_args, num_processes, main_process_port)
        else:
            _run_module_directly(module_path, remaining_args)
    else:
        parser.print_help()
        sys.exit(1)

