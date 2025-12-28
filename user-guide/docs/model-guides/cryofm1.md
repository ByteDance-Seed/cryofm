# CryoFM1 User Guide

> **CryoFM: A Flow-based Foundation Model for Cryo-EM Densities**  
> *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025.

## Overview

CryoFM is a flow-based foundation model designed for generating and working with 3D cryo-electron microscopy (cryo-EM) density maps. The model employs a Hierarchical Diffusion Transformer (HDiT) architecture to learn deep priors of 3D cryo-EM densities, enabling various downstream tasks including density map denoising, anisotropy correction, and missing wedge inpainting.

CryoFM1 is available in two variants:

- **CryoFM-S**: A smaller model optimized for 64×64×64 voxel volumes at 1.5 Å/pixel resolution
<figure markdown>
  ![CryoFM-S arch.](../images/cryofm1_arch-s.jpg)
  <figcaption>CryoFM-S architecture.</figcaption>
</figure>

- **CryoFM-L**: A larger model designed for 128×128×128 voxel volumes at 3.0 Å/pixel resolution
<figure markdown>
  ![CryoFM-L arch.](../images/cryofm1_arch-l.jpg)
  <figcaption>CryoFM-L architecture.</figcaption>
</figure>

## Prerequisites

Before using CryoFM1, ensure you have:

- The `cryofm` package installed (see [Installation Guide](../getting-started/installation.md))
- Model checkpoints and configuration files downloaded

## Basic Usage

### CryoFM-S: Unconditional Generation

CryoFM-S generates 64×64×64 voxel density maps at 1.5 Å/pixel resolution. Example outputs are shown below:

<figure markdown>
  ![CryoFM-S sampling examples.](../images/cryofm1_1-5apix.jpg)
  <figcaption>CryoFM-S sampling examples at 1.5 Å/pixel resolution.</figcaption>
</figure>

```python
import torch
from mmengine import Config
from cryofm.core.utils.mrc_io import save_mrc
from cryofm.projects.cryofm1.lit_modules import CryoFM1
from cryofm.core.utils.sampling_fm import sample_from_fm

# Load configuration and model
cfg = Config.fromfile("path_to/cryofm-v1/cryofm-s/config.yaml")
lit_model = CryoFM1.load_from_safetensors(
    "path_to/cryofm-v1/cryofm-s/model.safetensors", 
    cfg=cfg
)

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_model = lit_model.to(device)
lit_model.eval()

# Define vector field function for flow matching
def v_xt_t(_xt, _t):
    return lit_model(_xt, _t)

# Generate samples
# Note: Enable bfloat16 if your GPU supports it for better performance
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = sample_from_fm(
        v_xt_t, 
        lit_model.noise_scheduler, 
        method="euler", 
        num_steps=200, 
        num_samples=3, 
        device=device, 
        side_shape=64
    )
    # Apply z-scaling normalization if configured
    if hasattr(lit_model.cfg, "z_scale") and lit_model.cfg.z_scale.mean is not None:
        out = out * lit_model.cfg.z_scale.std + lit_model.cfg.z_scale.mean

# Save generated density maps
for i in range(3):
    save_mrc(
        out[i].float().cpu().numpy(), 
        f"sample-{i}.mrc", 
        apix=1.5  # Angstroms per pixel
    )
```

### CryoFM-L: Unconditional Generation

CryoFM-L generates 128×128×128 voxel density maps at 3.0 Å/pixel resolution. Example outputs are shown below:

<figure markdown>
  ![CryoFM-L sampling examples.](../images/cryofm1_3apix.jpg)
  <figcaption>CryoFM-L sampling examples at 3.0 Å/pixel resolution.</figcaption>
</figure>

```python
import torch
from mmengine import Config
from cryofm.core.utils.mrc_io import save_mrc
from cryofm.projects.cryofm1.lit_modules import CryoFM1
from cryofm.core.utils.sampling_fm import sample_from_fm

# Load configuration and model
cfg = Config.fromfile("path_to/cryofm-v1/cryofm-l/config.yaml")
lit_model = CryoFM1.load_from_safetensors(
    "path_to/cryofm-v1/cryofm-l/model.safetensors", 
    cfg=cfg
)

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_model = lit_model.to(device)
lit_model.eval()

# Define vector field function for flow matching
def v_xt_t(_xt, _t):
    return lit_model(_xt, _t)

# Generate samples
# Note: Enable bfloat16 if your GPU supports it for better performance
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = sample_from_fm(
        v_xt_t, 
        lit_model.noise_scheduler, 
        method="euler", 
        num_steps=200, 
        num_samples=3, 
        device=device, 
        side_shape=128
    )
    # Apply z-scaling normalization if configured
    if hasattr(lit_model.cfg, "z_scale") and lit_model.cfg.z_scale.mean is not None:
        out = out * lit_model.cfg.z_scale.std + lit_model.cfg.z_scale.mean

# Save generated density maps
for i in range(3):
    save_mrc(
        out[i].float().cpu().numpy(), 
        f"sample-{i}.mrc", 
        apix=3.0  # Angstroms per pixel
    )
```

## Advanced Usage

### Sampling Methods

The `sample_from_fm` function supports multiple ODE solvers for the flow matching process:

- **`"euler"`**: Euler method (default, fastest)
- **`"rk4"`**: 4th-order Runge-Kutta method (more accurate, slower)
- **`"midpoint"`**: Midpoint method (balanced)
- **`"heun"`**: Heun's method
- **`"ralston"`**: Ralston's method

For most use cases, `"euler"` with 200 steps provides a good balance between quality and speed. For higher quality, consider using `"rk4"` or increasing `num_steps`.

### Adjusting Sampling Parameters

- **`num_steps`**: Number of integration steps (default: 200). More steps generally yield better quality but take longer.
- **`num_samples`**: Number of samples to generate in a single batch.
- **`side_shape`**: Spatial dimensions of the output volume (64 for CryoFM-S, 128 for CryoFM-L).

## Synthetic Downstream Tasks

CryoFM1 can be used for various downstream tasks including denoising, anisotropy correction, and missing wedge restoration through Diffusion Posterior Sampling (DPS). This section describes how to prepare test datasets and run these tasks.

### Create Our Test Set

To run downstream tasks, you need to prepare a test dataset with the following structure:

```
path_to/cryofm1_1-5apix_dataset/
├── split/
│   └── test.csv
├── EMD-12042
└── ...
```

### Run Downstream Tasks

The `test_cryofm1.py` script performs downstream tasks using Flow Posterior Sampling. It processes each test sample by:

1. Applying a forward degradation operator (noise, missing wedge, etc.)
2. Generating two degraded half-maps
3. Estimating degradation operators from the half-map FSC
4. Running DPS to restore the clean density map
5. Computing FSC metrics between restored and ground truth maps

**Common Parameters:**

- `--data-root`: Path to your test dataset directory
- `--model-dir`: Path to the model directory (e.g., `path_to/cryofm-v1/cryofm-s/`)
- `--exp-name`: Experiment name for organizing output files
- `--num-timesteps`: Number of DPS sampling steps (default: 200, higher values improve quality but are slower)
- `--task-names`: Task type(s) to perform
- `--eval-n-samples`: Number of samples to evaluate (default: all samples in test set)

**Output:**
Results are saved in `work_dirs/{exp_name}/{map_id}/` for each test sample, including:

- Restored density maps (`new_y1.mrc`, `new_y2.mrc`, `new_y_avg.mrc`)
- FSC curves comparing restored maps with ground truth
- Statistics and metrics in the log file

#### Spectral Noise Denoising

Removes isotropic spectral noise from cryo-EM density maps. The noise power spectrum is estimated from FSC curves based on the specified SNR index.

```bash
python scripts/test_cryofm1.py \
    --data-root path_to/cryofm1_1-5apix_dataset/ \
    --model-dir path_to/cryofm-v1/cryofm-s/ \
    --exp-name cryofm_sn_snr1 \
    --num-timesteps 1000 \
    --task-names spectral_noise \
    --snr-idx 1
```

**Parameters:**

- `--snr-idx`: SNR index (1-5) corresponding to different FSC-based SNR levels. Higher indices indicate higher noise levels.

#### Anisotropic Noise Denoising

Removes anisotropic spectral noise, which is common in cryo-ET data due to limited tilt angles. The noise pattern depends on the tilt angle and amplification factor.

```bash
python scripts/test_cryofm1.py \
    --data-root path_to/cryofm1_1-5apix_dataset/ \
    --model-dir path_to/cryofm-v1/cryofm-s/ \
    --exp-name cryofm_asn_snr1_tilt15 \
    --num-timesteps 1000 \
    --task-names anisotropic_spectral_noise \
    --snr-idx 1 \
    --tilt-angle 15
```

**Parameters:**

- `--snr-idx`: SNR index (1-5) for the base noise level
- `--tilt-angle`: Maximum tilt angle in degrees (e.g., 15, 30, 45, 60). Larger angles result in less anisotropic noise.

#### Missing Wedge Restoration

Restores information lost due to the missing wedge in cryo-ET reconstructions. The missing wedge is defined by the maximum tilt angle.

```bash
python scripts/test_cryofm1.py \
    --data-root path_to/cryofm1_1-5apix_dataset/ \
    --model-dir path_to/cryofm-v1/cryofm-s/ \
    --exp-name cryofm_mw_tilt60 \
    --num-timesteps 1000 \
    --task-names missing_wedge \
    --tilt-angle 60
```

**Parameters:**

- `--tilt-angle`: Maximum tilt angle in degrees. The missing wedge corresponds to the unobserved region in Fourier space beyond this angle.

**Tips:**

- For better quality, increase `--num-timesteps` (e.g., 1000-2000 steps), though this will take longer
- You can adjust DPS hyperparameters with `--lamb-base` and `--lamb-w-max` if needed
- Check the FSC curves in the output directory to evaluate restoration quality


## Citation

If you use CryoFM1 in your research, please cite:

```bibtex
@inproceedings{
zhou2025cryofm,
title={Cryo{FM}: A Flow-based Foundation Model for Cryo-{EM} Densities},
author={Yi Zhou and Yilai Li and Jing Yuan and Quanquan Gu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=T4sMzjy7fO}
}
```

