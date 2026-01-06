# CryoFM2: Quick Start

## Prerequisites

Before using CryoFM2, please ensure:

1. **Install CryoFM**: Follow the [Installation Guide](../../getting-started/installation.md) to install `cryofm`.
2. **Download Model Weights**: CryoFM2 model weights are available for download from [Hugging Face](https://huggingface.co/ByteDance-Seed/cryofm-v2).

---

## Play with CryoFM2

### Standalone map modification

CryoFM2 supports various density map processing tasks through different forward operators. The `--op` parameter specifies the forward operator that defines the inverse problem formulation, enabling different processing capabilities such as denoising, anisotropy correction, non-uniform refinement and different style post-processing. For detailed information about available operators, see the [Operators Guide](operators.md).

#### Denoising a density map

The `denoise` command name encompasses multiple processing capabilities including denoising, anisotropy correction, and non-uniform refinement. All these functions are provided by the pre-trained model (`cryofm2-pretrain`). The specific operation is controlled by the `--op` parameter. Always use `--bf16` if possible.

Remove noise from a pair of half maps using the `denoise` operator:

```bash
# Single GPU
cfm denoise -i1 half_map_1.mrc -i2 half_map_2.mrc -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op denoise --norm-grad --use-lamb-w
```

#### Anisotropy correction

Correct anisotropy in a density map using the `inpaint denoise` operator:

```bash
# Single GPU
cfm denoise -i1 half_map_1.mrc -i2 half_map_2.mrc -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op inpaint denoise --data-starfile-path path/to/particles.star \
    --op denoise --norm-grad --use-lamb-w
```

#### Non-uniform refinement

Perform non-uniform refinement using the `non-uniform` operator:

```bash
# Single GPU
cfm denoise -i1 half_map_1.mrc -i2 half_map_2.mrc -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op non-uniform --norm-grad --use-lamb-w
```

#### Density map enhancement (EMhancer style)

Apply EMhancer-style enhancement using the `cryofm2-emhancer` model. You can use either a single input map or a pair of half maps with the `--op` parameter for control:

```bash
# Enhance a single input map
cfm enhance -i input_map.mrc -o ./output_emhancer \
    --model-dir path/to/cryofm-v2/cryofm2-emhancer --output-tag 1

# Add extra control
cfm denoise -i input_map.mrc -i1 half_map_1.mrc -i2 half_map_2.mrc -o ./output_emhancer \
    --model-dir path/to/cryofm-v2/cryofm2-emhancer --op denoise --norm-grad --use-lamb-w --output-tag 1
```

#### Density map enhancement (EMReady style)

Apply EMReady-style enhancement using the `cryofm2-emready` model. You can use either a single input map or a pair of half maps with the `--op` parameter for control:

```bash
# Enhance a single input map
cfm enhance -i input_map.mrc -o ./output_emready \
    --model-dir path/to/cryofm-v2/cryofm2-emready --output-tag 0 --cfg-weight 0.5

# Add extra control
cfm denoise -i input_map.mrc -i1 half_map_1.mrc -i2 half_map_2.mrc -o ./output_emready \
    --model-dir path/to/cryofm-v2/cryofm2-emready --op denoise --norm-grad --use-lamb-w --output-tag 0 --cfg-weight 0.5
```

### Multi-GPU processing

All CryoFM2 commands support multi-GPU processing through the `--num_processes` parameter. Simply specify the number of GPUs you want to use:

```bash
# Example: Using 4 GPUs for denoising
cfm denoise --num_processes 4 -i1 half_map_1.mrc -i2 half_map_2.mrc -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain \
    --op denoise --norm-grad --use-lamb-w
```

The CLI commands automatically handle the accelerate multi-GPU setup when `--num_processes` is specified. For more advanced control, you can also use `accelerate launch` directly:

```bash
accelerate launch --num_processes 4 cfm denoise -i1 half_map_1.mrc -i2 half_map_2.mrc -o ./output \
    --model-dir path/to/cryofm-v2/cryofm2-pretrain --op denoise
```

### Use in RELION

TBA
