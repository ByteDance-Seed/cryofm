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
- Triton, optional, for accelerated backend kernels

### Install from source

```bash
conda create -n cryoseed python=3.10 -y
conda activate cryoseed
python -m pip install --upgrade pip

# Install PyTorch first according to your CUDA / platform environment.
python -m pip install torch

# Install runtime dependencies used by cryoSeed.
python -m pip install numpy scipy pandas matplotlib mrcfile starfile pyyaml tqdm

# Optional: install Triton for accelerated backend kernels.
python -m pip install triton

# Install in editable mode from the repository root.
python -m pip install -e .
```

## Getting Started

Apart from using as an reconstruction backbone in cryoFM, cryoSeed can also be used independently at different levels depending on whether you want to run an existing workflow or reuse individual reconstruction components.

### 1. Construct a reconstruction pipeline in Python

The most flexible usage mode is to assemble a reconstruction workflow directly in Python. This is mainly intended for method development, debugging, and experiments that require access to individual reconstruction components.

```python
from cryoseed.config import MainConfig
from cryoseed.data import ParticleDataset, build_half_dataloaders
from cryoseed.fft.fft_torch import primal_to_fourier_3d
from cryoseed.modules.pose import Pose
from cryoseed.modules.statistics import NoiseVariance, PriorVariance
from cryoseed.modules.volume import VoxelGrid
from cryoseed.metrics.fsc import calc_fsc, fsc_to_resolution
from cryoseed.optim import EMSolver
from cryoseed.optim.pose import PoseSearcher
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
        switch_to_euler_order=4,
    ):
        self.state = state
        self.side_length_step = int(side_length_step)
        self.max_side_length = None if max_side_length is None else int(max_side_length)
        self.healpix_step_every = int(healpix_step_every)
        self.switch_to_euler_order = int(switch_to_euler_order)

    def step(self):
        next_side_length = int(self.state.schedule.side_length) + self.side_length_step
        if self.max_side_length is not None:
            next_side_length = min(next_side_length, self.max_side_length)
        self.state.schedule.side_length = next_side_length

        if (self.state.progress.epoch + 1) % self.healpix_step_every == 0:
            self.state.schedule.healpix_order += 1

        if self.state.schedule.healpix_order >= self.switch_to_euler_order:
            self.state.schedule.pose_search_scope = "local"
            self.state.schedule.pose_search_strategy = "euler"
        else:
            self.state.schedule.pose_search_scope = "global"
            self.state.schedule.pose_search_strategy = "healpix"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # This demo shows a minimal cryo-EM homogeneous refinement loop based on EM.
    config = MainConfig.from_file("minimal_config.yaml")

    # Build the particle dataset and split it into two halves for gold-standard refinement.
    dataset = ParticleDataset(
        star_path=config.io.star_path,
        data_prefix=config.io.data_path,
        num_particles=config.data.num_particles,
        image_size=config.data.image_size,
        angpix=config.data.angpix,
        default_optic_params=config.data.default_optic_params,
        default_particle_params=config.data.default_particle_params,
    )
    dataset.populate_data_config(config.data)

    dl_half0, dl_half1, _, _ = build_half_dataloaders(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        device=device,
        seed=42,
        drop_last=False,
    )

    # The shared state stores the current search resolution / schedule for both halves.
    state = OptimState()
    state.schedule.healpix_order = int(config.pose_search.init_healpix_order)
    state.schedule.proj_cache_backend = "memory" if config.scheduler.use_cache else "none"
    state.schedule.side_length = max(
        8,
        2 * int(config.data.image_size * config.data.angpix / config.refinement.init_lowpass_angstrom),
    )

    # Each half keeps its own volume, noise model, and pose estimates.
    volume_half0 = VoxelGrid(
        grid_size=config.data.image_size,
        device=device,
    )
    volume_half1 = VoxelGrid(
        grid_size=config.data.image_size,
        device=device,
    )
    noise_half0 = NoiseVariance(
        image_size=config.data.image_size,
        device=device,
    )
    noise_half1 = NoiseVariance(
        image_size=config.data.image_size,
        device=device,
    )
    prior = None
    if config.statistics.use_prior:
        prior = PriorVariance(
            image_size=config.data.image_size,
            device=device,
        )
    pose_half0 = Pose(
        num_particles=config.data.num_particles,
        device=device,
    )
    pose_half1 = Pose(
        num_particles=config.data.num_particles,
        device=device,
    )

    with mrcfile.open(config.io.ref_volume_path, permissive=True) as mrc:
        ref_volume = torch.tensor(mrc.data, device=device).unsqueeze(0)
    ref_volume_fourier = primal_to_fourier_3d(ref_volume)

    # Both halves start from the same reference volume.
    volume_half0.load_volume(ref_volume_fourier)
    volume_half1.load_volume(ref_volume_fourier)

    # Initialize the optional statistics modules before entering the EM loop.
    if config.statistics.use_noise:
        noise_half0.from_data([dl_half0, dl_half1])
    if config.statistics.use_noise:
        noise_half1.load_state_dict(noise_half0.state_dict())
    if prior is not None:
        prior.from_volume(ref_volume_fourier)

    searcher_half0 = PoseSearcher(
        state=state,
        volume=volume_half0,
        noise=noise_half0,
        pose=pose_half0,
        config=config,
        device=device,
    )
    searcher_half1 = PoseSearcher(
        state=state,
        volume=volume_half1,
        noise=noise_half1,
        pose=pose_half1,
        config=config,
        device=device,
    )
    solver_half0 = EMSolver(state=state, pose_searcher=searcher_half0, prior=prior)
    solver_half1 = EMSolver(state=state, pose_searcher=searcher_half1, prior=prior)
    scheduler = NaiveScheduler(
        state,
        side_length_step=2,
        max_side_length=config.data.image_size,
        healpix_step_every=2,
        switch_to_euler_order=4,
    )

    for epoch in range(config.refinement.num_epochs):
        state.progress.epoch = epoch
        state.metrics.confidence_sum = 0.0
        state.metrics.confidence_count = 0

        # Reset accumulation buffers for the new EM iteration.
        solver_half0.zero_accum()
        solver_half1.zero_accum()

        # Half 0: E-step, pose search, followed by M-step, volume / noise / pose update.
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
        state.metrics.fsc_scores = torch.as_tensor(fsc_scores, dtype=torch.float32, device=device)
        state.metrics.fsc_resolution = fsc_to_resolution(
            fsc_scores,
            fsc_freqs,
            threshold=config.refinement.fsc_threshold,
            voxel_size=config.data.angpix,
        )
        scheduler.step()

        print(
            f"epoch={epoch} "
            f"half0_shape={tuple(volume_half0.volume.shape)} "
            f"half1_shape={tuple(volume_half1.volume.shape)} "
            f"avg_confidence={state.metrics.avg_confidence:.4f} "
            f"resolution={state.metrics.fsc_resolution:.2f}A "
            f"next_L={state.schedule.side_length} "
            f"next_healpix={state.schedule.healpix_order} "
            f"next_strategy={state.schedule.pose_search_strategy}"
        )


if __name__ == "__main__":
    main()
```

### 2. Import individual modules

Individual cryoSeed modules can also be imported into custom scripts. This is useful when only a specific component is needed, such as a volume representation, Fourier transform utility, projection operation, or cryo-EM physics helper.

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

### 3. Use the command-line interface

cryoSeed also includes a configuration-driven command-line interface for running reconstruction experiments. This interface is mainly intended for reproducible experiments and internal benchmarking workflows.

The simplest way to launch homogeneous refinement is to pass the main input and output arguments directly:

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
├── cryoem/                # cryo-EM physics utilities, such as CTF and masks
├── backends/              # backend implementations with PyTorch / Triton
├── data/                  # data loading and preprocessing
```
