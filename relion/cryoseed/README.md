# cryoSeed

**cryoSeed** is a modular cryo-EM 3D reconstruction module included as part of this repository update. It provides PyTorch-native components for building and experimenting with 3D reconstruction workflows, with an emphasis on composability, differentiability, and integration with modern machine learning pipelines.

cryoSeed is intended to expose core reconstruction functionality in a form that can be reused, modified, and extended within this codebase. Its components cover common elements of cryo-EM reconstruction, including data loading, pose searching, Fourier-domain volume representation, projection operations, noise and prior statistics, EM-style optimization, and evaluation utilities such as FSC. The module is designed to support method development and prototyping, especially for experiments that connect cryo-EM reconstruction with AI-based priors, like cryoFM, with `./scripts/cryoseed_wrapper.py`. 

---

## Highlights

- PyTorch-native implementation of core reconstruction components
- Differentiable operators suitable for gradient-based workflows
- Modular components for reconstruction, pose search, statistics, and optimization
- GPU-oriented execution with optional accelerated backends
- Configuration-driven homogeneous refinement examples
- Research-oriented design for prototyping and benchmarking reconstruction methods

---

## Installation

### Requirements

- Python >= 3.10
- PyTorch
- CUDA-enabled GPU recommended
- Triton (for accelerated backend kernels)

### Install from source

```bash
conda create -n cryoseed python=3.10 -y
conda activate cryoseed
python -m pip install --upgrade pip
```

We recommend installing PyTorch manually first so that you can select the
appropriate build for your CUDA, ROCm, or CPU environment. See the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/)
for the command matching your platform.

Then install cryoSeed from the repository. The remaining runtime dependencies
declared in `pyproject.toml` will be installed automatically.

```bash
python -m pip install -e .
```

## Getting Started

cryoSeed supports multiple usage modes depending on the level of control and customization you need.

### 1. Construct your own reconstruction pipeline

The most flexible way to use cryoSeed is to build a reconstruction workflow directly in Python. This mode is intended for advanced users and method developers who want fine-grained control over the reconstruction procedure and its individual components.

The example below keeps the pipeline explicit in Python rather than loading an external YAML file. Core objects are initialized directly from named variables, and the EM loop uses `HEALPixPoseSearcher` directly.

```python
from cryoseed.data import ParticleDataset, build_half_dataloaders
from cryoseed.fft.fft_torch import primal_to_fourier_3d
from cryoseed.modules.pose import Pose
from cryoseed.modules.statistics import NoiseVariance, PriorVariance
from cryoseed.modules.volume import VoxelGrid
from cryoseed.metrics.fsc import calc_fsc, fsc_to_resolution
from cryoseed.optim import EMSolver
from cryoseed.optim.pose import HEALPixPoseSearcher
from cryoseed.state import OptimState

import mrcfile
import torch


class NaiveScheduler:
    def __init__(
        self,
        state,
        *,
        side_length_step=2,
        max_side_length=None,
        healpix_step_every=2,
    ):
        self.state = state
        self.side_length_step = int(side_length_step)
        self.max_side_length = None if max_side_length is None else int(max_side_length)
        self.healpix_step_every = int(healpix_step_every)

    def step(self):
        next_side_length = int(self.state.schedule.side_length) + self.side_length_step
        if self.max_side_length is not None:
            next_side_length = min(next_side_length, self.max_side_length)
        self.state.schedule.side_length = next_side_length

        if (self.state.progress.epoch + 1) % self.healpix_step_every == 0:
            self.state.schedule.healpix_order += 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # This demo shows a minimal cryo-EM homogeneous refinement loop based on EM.
    star_path = "path/to/particles.star"
    data_path = "path/to/particle_stack_dir"
    ref_volume_path = "path/to/reference_volume.mrc"

    image_size = 256
    angpix = 1.5
    num_particles = 100_000
    batch_size = 8
    num_workers = 0
    num_epochs = 5

    init_lowpass_angstrom = 20.0
    init_healpix_order = 3
    trans_grid_samples = 5
    use_projection_cache = False
    noise_enabled = True
    prior_enabled = True

    default_optic_params = {
        "voltage_kv": 300.0,
        "spherical_aberration_mm": 2.7,
        "amplitude_contrast": 0.1,
        "ctf_bfactor": 0.0,
        "ctf_scale": 1.0,
        "phase_shift_deg": 0.0,
    }
    default_particle_params = {
        "defocus_u_angstrom": 15_000.0,
        "defocus_v_angstrom": 15_000.0,
        "defocus_angle_deg": 0.0,
    }

    # Build the particle dataset and split it into two halves for gold-standard refinement.
    dataset = ParticleDataset(
        star_path=star_path,
        data_prefix=data_path,
        num_particles=num_particles,
        image_size=image_size,
        angpix=angpix,
        default_optic_params=default_optic_params,
        default_particle_params=default_particle_params,
    )

    dl_half0, dl_half1, _, _ = build_half_dataloaders(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
        seed=42,
        drop_last=False,
    )

    # The shared state stores the current search resolution / schedule for both halves.
    state = OptimState()
    state.schedule.healpix_order = init_healpix_order
    state.schedule.trans_grid_samples = trans_grid_samples
    state.schedule.proj_cache_backend = (
        "memory" if use_projection_cache else "none"
    )
    state.schedule.side_length = max(
        8,
        2 * int(image_size * angpix / init_lowpass_angstrom),
    )

    # Each half keeps its own volume, noise model, and pose estimates.
    volume_half0 = VoxelGrid(grid_size=image_size, device=device)
    volume_half1 = VoxelGrid(grid_size=image_size, device=device)
    noise_half0 = NoiseVariance(image_size=image_size, device=device) if noise_enabled else None
    noise_half1 = NoiseVariance(image_size=image_size, device=device) if noise_enabled else None
    prior = PriorVariance(image_size=image_size, device=device) if prior_enabled else None
    pose_half0 = Pose(num_particles=num_particles, device=device)
    pose_half1 = Pose(num_particles=num_particles, device=device)

    with mrcfile.open(ref_volume_path, permissive=True) as mrc:
        ref_volume = torch.tensor(mrc.data, device=device).unsqueeze(0)
    ref_volume_fourier = primal_to_fourier_3d(ref_volume)

    # Both halves start from the same reference volume.
    volume_half0.load_volume(ref_volume_fourier)
    volume_half1.load_volume(ref_volume_fourier)

    # Initialize the optional statistics modules before entering the EM loop.
    if noise_half0 is not None:
        noise_half0.from_data([dl_half0, dl_half1])
    if noise_half1 is not None and noise_half0 is not None:
        noise_half1.load_state_dict(noise_half0.state_dict())
    if prior is not None:
        prior.from_volume(ref_volume_fourier)

    searcher_half0 = HEALPixPoseSearcher(
        state=state,
        volume=volume_half0,
        noise=noise_half0,
        pose=pose_half0,
        device=device,
        trans_grid_samples=trans_grid_samples,
    )
    searcher_half1 = HEALPixPoseSearcher(
        state=state,
        volume=volume_half1,
        noise=noise_half1,
        pose=pose_half1,
        device=device,
        trans_grid_samples=trans_grid_samples,
    )
    solver_half0 = EMSolver(state=state, pose_searcher=searcher_half0, prior=prior)
    solver_half1 = EMSolver(state=state, pose_searcher=searcher_half1, prior=prior)
    scheduler = NaiveScheduler(
        state,
        side_length_step=2,
        max_side_length=image_size,
        healpix_step_every=2,
    )

    for epoch in range(num_epochs):
        state.progress.epoch = epoch

        # Reset accumulation buffers for the new EM iteration.
        solver_half0.zero_accum()
        solver_half1.zero_accum()

        # Half 0: E-step (pose search) followed by M-step (volume / noise / pose update).
        state.progress.half = 0
        solver_half0.refresh()
        for batch in dl_half0:
            batch = batch.to(device, non_blocking=True)
            result = solver_half0.infer(batch)
            solver_half0.accumulate(result)
        solver_half0.update()

        # Half 1 repeats the same EM update independently.
        state.progress.half = 1
        solver_half1.refresh()
        for batch in dl_half1:
            batch = batch.to(device, non_blocking=True)
            result = solver_half1.infer(batch)
            solver_half1.accumulate(result)
        solver_half1.update()

        # Evaluate the two half maps with FSC and update the schedule for the next epoch.
        vol0 = volume_half0.volume_real[0].detach().cpu().numpy()
        vol1 = volume_half1.volume_real[0].detach().cpu().numpy()
        fsc_scores, fsc_freqs = calc_fsc(vol0, vol1)
        state.homorefine.metrics.fsc_scores = torch.as_tensor(
            fsc_scores,
            dtype=torch.float32,
            device=device,
        )
        state.homorefine.metrics.fsc_resolution = fsc_to_resolution(
            fsc_scores,
            fsc_freqs,
            threshold=0.143,
            angpix=angpix,
        )
        scheduler.step()

        print(
            f"epoch={epoch} "
            f"half0_shape={tuple(volume_half0.volume.shape)} "
            f"half1_shape={tuple(volume_half1.volume.shape)} "
            f"avg_confidence={state.homorefine.metrics.avg_confidence:.4f} "
            f"resolution={state.homorefine.metrics.fsc_resolution:.2f}A "
            f"next_L={state.schedule.side_length} "
            f"next_healpix={state.schedule.healpix_order}"
        )


if __name__ == "__main__":
    main()
```

### 2. Import individual modules

If you only need specific components from cryoSeed, you can directly import individual modules into your own code. This is useful when only part of the framework is needed, such as a projection operator, a reconstruction loss, or a physics-related utility.

```python
import mrcfile
import torch

from cryoseed.cryoem.rotation import euler_to_matrix
from cryoseed.fft.fft_torch import fourier_to_primal_2d, primal_to_fourier_3d
from cryoseed.modules.volume import VoxelGrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ref_volume_path = "path/to/reference_volume.mrc"

# Load a real-space 3D map and infer the voxel grid size from the map shape.
with mrcfile.open(ref_volume_path, permissive=True) as mrc:
    volume_real = torch.tensor(mrc.data, dtype=torch.float32, device=device).unsqueeze(0)
grid_size = volume_real.shape[-1]

# Create a single-volume voxel representation and import the map into Fourier space.
voxel = VoxelGrid(grid_size=grid_size, device=device)
voxel.load_volume(primal_to_fourier_3d(volume_real))

# Generate one random pose with ZYZ Euler angles.
euler = torch.rand(1, 1, 3, device=device)
euler[..., 0] *= 2 * torch.pi
euler[..., 1] *= torch.pi
euler[..., 2] *= 2 * torch.pi
rotation = euler_to_matrix(euler)

# Project the 3D volume to a 2D Fourier slice.
projection_fourier = voxel.project(rotation=rotation, side_length=grid_size).squeeze(0)

# Convert the Fourier projection back to real space for visualization.
projection_real = fourier_to_primal_2d(projection_fourier).real

print(f"voxel_shape={tuple(voxel.volume.shape)}")
print(f"projection_shape={tuple(projection_real.shape)}")
```

### 3. Use the command-line interface (CLI)

For end users and routine experiments, cryoSeed provides a command-line interface driven by configuration files. This is the most convenient way to run complete reconstruction experiments without writing Python code. Users can specify datasets, reconstruction settings, and runtime options through configuration files, and launch end-to-end workflows directly from the command line.

The two top-level workflow commands are:

- `cryoseed homorefine`: homogeneous refinement
- `cryoseed abinitio`: ab initio reconstruction

The bundled config templates serve different roles:

- `full_config.yaml`: full schema reference with schema-level defaults for every field
- `src/cryoseed/config/defaults/homorefine.yaml`: homorefine template listing only homorefine-supported fields
- `src/cryoseed/config/defaults/abinitio.yaml`: ab initio template listing only ab initio-supported fields

The simplest way to launch homogeneous refinement is to pass the main input / output arguments directly:

```bash
cryoseed homorefine \
  --star-path path/to/particles.star \
  --data-path path/to/particle_stack_dir \
  --ref-volume-path path/to/reference_volume.mrc \
  --output-path outputs/demo \
  --num-epochs 5 \
  --batch-size 8
```

For distributed execution, launch the same command with `torchrun` and provide a config file:

```bash
torchrun --nproc_per_node=NPROC_PER_NODE \
  -m cryoseed homorefine \
  --config minimal_config.yaml
```

For a complete list of CLI flags and configuration overrides, see:

```bash
cryoseed homorefine --help
```

The ab initio command is launched in the same way:

```bash
cryoseed abinitio \
  --star-path path/to/particles.star \
  --data-path path/to/particle_stack_dir \
  --output-path outputs/abinitio_demo \
  --num-volumes 3 \
  --num-epochs 10
```

Use the command-specific help to see only the overrides supported by that workflow:

```bash
cryoseed abinitio --help
cryoseed homorefine --help
```

When a run starts, the saved `output_path/config.yml` snapshot is also trimmed to the current command's supported fields.

## Project Structure

The current organization of cryoSeed follows a layered and modular design:

```text
cryoSeed/
├── cli.py                 # command-line entry point
├── config/                # experiment and runtime configurations
├── engines/               # execution engines for reconstruction workflows
├── state/                 # runtime and optimization state management
├── modules/               # high-level reconstruction modules
├── optim/                 # solvers and optimization logic
├── ops/                   # core operators such as projection and backprojection
├── cryoem/                # cryo-EM physics utilities (e.g., CTF, masks)
├── backends/              # backend implementations with PyTorch / Triton
├── data/                  # data loading and preprocessing
└── docs/                  # documentation and usage guides
```
