# CryoFM2: Likelihood Control

## Overview

CryoFM2 uses the Flow-based Posterior Sampling (FPS) framework to combine the pretrained model as a Bayesian prior with task-specific likelihood functions for density map processing. This document is intended for developers and explains how to optimize processing results for different tasks by tuning likelihood control parameters.

## Flow-based Posterior Sampling (FPS) Principles

FPS is the core inference framework of CryoFM2, which combines the generative model prior with the likelihood function of observed data:

$$\mathbf{v}_t = \mathbf{v}_t^{\text{prior}} + \lambda \nabla_{\mathbf{x}_t} \log p(\mathbf{y} | \mathbf{x}_0(\mathbf{x}_t, t))$$

where:

- $\mathbf{v}_t^{\text{prior}}$ is the model-predicted velocity field (prior)
- $\lambda$ is the weight of likelihood guidance
- $\nabla_{\mathbf{x}_t} \log p(\mathbf{y} | \mathbf{x}_0)$ is the gradient of the likelihood function (data consistency constraint)

By adjusting $\lambda$ and gradient normalization strategies, you can balance the model prior and data consistency to achieve optimal processing results for different tasks.

## Core Parameters

### Likelihood Guidance Parameters

#### `--lamb-base` (default: 1000.0)

Controls the base step size for likelihood guidance. This is one of the most important parameters, directly affecting the balance between data consistency and prior.

#### `--use-lamb-w` (default: False)

Enables a decayed lambda scheduler. When enabled, the lambda weight decays over timesteps:

$$\lambda_w(t) = \min\left(\frac{t}{1000 - t}, \lambda_{\max}\right)$$

This allows stronger data constraints in early sampling steps (high noise) and more reliance on the model prior in later steps (low noise).

#### `--lamb-w-max` (default: 5.0)

When `--use-lamb-w` is enabled, sets the maximum value of the lambda weight. Controls the maximum guidance strength in early timesteps.

#### `--norm-grad` (default: False, **strongly recommended**)

Normalizes the likelihood gradient. This is a key feature of the FPS framework, ensuring that the gradient direction rather than magnitude affects the sampling process.

**Important:** For flow matching models, `--norm-grad` must be enabled. Disabling it will cause unstable sampling.

Enable with:
```bash
--norm-grad
```

### Sampling Parameters

#### `--num-timesteps` (default: 200)

Number of sampling timesteps. More timesteps usually yield better results but increase computation time.

- **Quick preview**: 50-100 steps
- **Standard quality**: 200 steps (recommended)
- **High quality**: 300-500 steps

#### `--odeint` (default: "euler")

ODE solver method. Currently supports the "euler" method.

### Operator-Specific Parameters

#### `--fmask-threshold` (default: 10.0)

Fourier mask threshold for the `inpaint` operator. Controls which frequency regions are considered missing.

#### `--nbands` (default: 64)

Number of wavelet bands for the `non-uniform` operator. Controls the frequency decomposition granularity for non-uniform denoising.

## Parameter Tuning Strategies for Different Tasks

### Denoising Task (`--op denoise`)

**Recommended configuration:**

```bash
python -m cryofm.projects.cryofm2.uncond_sampling \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op denoise \
    --norm-grad \
    --use-lamb-w \
    --lamb-base 1000.0 \
    --lamb-w-max 5.0 \
    --num-timesteps 200 \
    --bf16
```

### Missing Wedge Inpainting Task (`--op inpaint`)

**Recommended configuration:**

```bash
python -m cryofm.projects.cryofm2.uncond_sampling \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op inpaint \
    --data-starfile-path path/to/relion_data.star \
    --norm-grad \
    --use-lamb-w \
    --lamb-base 1000.0 \
    --lamb-w-max 5.0 \
    --fmask-threshold 10.0 \
    --num-timesteps 200 \
    --bf16
```

### Combined Denoising and Inpainting Task (`--op denoise inpaint`)

**Recommended configuration:**

```bash
python -m cryofm.projects.cryofm2.uncond_sampling \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op denoise inpaint \
    --data-starfile-path path/to/relion_data.star \
    --norm-grad \
    --use-lamb-w \
    --lamb-base 1000.0 \
    --lamb-w-max 5.0 \
    --fmask-threshold 10.0 \
    --num-timesteps 200 \
    --bf16
```

### Non-uniform Denoising Task (`--op non-uniform`)

**Recommended configuration:**

```bash
python -m cryofm.projects.cryofm2.uncond_sampling \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op non-uniform \
    --norm-grad \
    --use-lamb-w \
    --lamb-base 1000.0 \
    --nbands 64 \
    --num-timesteps 200 \
    --bf16
```

### Conditional Generation Tasks (EMhancer/EMReady)

**Recommended configuration:**

```bash
python -m cryofm.projects.cryofm2.cond_sampling \
    -i input_map.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-emhancer \
    --output-tag 1 \
    --cfg-weight 2.0 \
    --num-timesteps 200
```

**Parameter adjustment suggestions:**

- **EMhancer style** (`--output-tag 1`):
  - `--cfg-weight`: 1.5-3.0 (default 2.0)

- **EMReady style** (`--output-tag 0`):
  - `--cfg-weight`: 0.3-0.7 (default 0.5)

**Using with operators:**

```bash
python -m cryofm.projects.cryofm2.cond_sampling \
    -i input_map.mrc \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-emhancer \
    --output-tag 1 \
    --op denoise \
    --norm-grad \
    --use-lamb-w \
    --lamb-base 1000.0 \
    --cfg-weight 2.0
```

## Advanced Usage

### Performance Optimization

#### Multi-GPU Inference

Use `accelerate launch` for multi-GPU inference:

```bash
NCCL_DEBUG=ERROR accelerate launch \
    --num_processes=4 \
    --main_process_port=8881 \
    python -m cryofm.projects.cryofm2.uncond_sampling \
        -i1 half_map_1.mrc \
        -i2 half_map_2.mrc \
        -o ./output \
        --model-dir path/to/cryofm-v2/cryofm2-pretrain \
        --op denoise \
        --norm-grad \
        --use-lamb-w
```

#### Mixed Precision Inference

Use the `--bf16` flag to reduce memory usage and speed up inference (requires GPU support for bfloat16):

```bash
python -m cryofm.projects.cryofm2.uncond_sampling \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op denoise \
    --norm-grad \
    --use-lamb-w \
    --bf16
```

#### Batch Size Adjustment

Adjust batch size based on GPU memory capacity:

```bash
python -m cryofm.projects.cryofm2.uncond_sampling \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op denoise \
    --norm-grad \
    --use-lamb-w \
    --batch-size 8 \
    --bf16
```

### Post-processing Options

#### FSC Weighting (`--fsc-weighting`)

Apply FSC weighting to the output to preserve high-frequency details:

```bash
python -m cryofm.projects.cryofm2.uncond_sampling \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op denoise \
    --norm-grad \
    --use-lamb-w \
    --fsc-weighting
```

#### Spectral Mixing (`--spectral-mixing`)

Use spectral mixing techniques to further optimize results (experimental feature):

```bash
python -m cryofm.projects.cryofm2.uncond_sampling \
    -i1 half_map_1.mrc \
    -i2 half_map_2.mrc \
    -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op denoise \
    --norm-grad \
    --use-lamb-w \
    --spectral-mixing
```

## References

- [Operators Guide](operators.md): Learn about how different operators work
- [Unconditional Sampling](unconditional-sampling.md): Learn how to use the Python API
