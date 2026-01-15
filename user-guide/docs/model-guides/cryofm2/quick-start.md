# CryoFM2: Quick Start

## Prerequisites

Before using CryoFM2, please ensure:

1. **Install CryoFM**: Follow the [Installation Guide](../../getting-started/installation.md) to install `cryofm`.
2. **Download Model Weights**: CryoFM2 model weights are available for download from [Hugging Face](https://huggingface.co/ByteDance-Seed/cryofm-v2). To download the model weights, first install the Hugging Face CLI tool:
   ```bash
   pip install huggingface-hub
   ```
   Then download all model files using:
   ```bash
   hf download ByteDance-Seed/cryofm-v2 --local-dir ./cryofm-v2
   ```
   This will download all necessary model files (including `cryofm2-pretrain`, `cryofm2-emhancer`, and `cryofm2-emready`) to the `./cryofm-v2` directory. You can change `./cryofm-v2` to your preferred download location.

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

The CLI commands automatically handle the accelerate multi-GPU setup when `--num_processes` is specified.

### Use in RELION

CryoFM2 can be integrated with RELION's refinement pipeline through the `--external_reconstruct` option. This allows RELION to use CryoFM2 for map reconstruction during the refinement process.

Here's an example script for running RELION refinement with CryoFM2:

```bash
# Set project paths
EXP_DIR=/path/to/relion/project
OUTPUT_DIR=Refine3D_cryofm2/job001
REF_MAP_PATH=reference_map.mrc
MASK_MAP_PATH=solvent_mask.mrc

# Set up environment variables, if you have 8 GPU, you should use --num_processes 8
export NCCL_DEBUG=ERROR
export CONDA_ENV="cryofm"
export CRYOFM_MODEL_DIR=/path/to/cryofm-v2/cryofm2-pretrain
export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="/path/to/conda/envs/cryofm/bin/python /path/to/cryofm/relion/relion_wrapper.py --mask-path ${MASK_MAP_PATH} --bbox --op denoise inpaint --fmask-threshold 10 --threshold-res 10 --norm-grad --use-lamb-w --skip-spectral-trailing --spectral-mixing"

cd ${EXP_DIR}
mkdir -p ${OUTPUT_DIR}

# Run RELION refinement with CryoFM2
mpirun  relion_refine_mpi \
  --o ${OUTPUT_DIR}/run \
  --auto_refine \
  --split_random_halves \
  --i particles.star \
  --ref ${REF_MAP_PATH} \
  ...
  --external_reconstruct |& tee ${OUTPUT_DIR}/console_log.txt
```

**Key parameters:**

- `CRYOFM_MODEL_DIR`: Path to the CryoFM2 model directory
- `RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE`: Command to launch the CryoFM2 wrapper with desired operators and options
- `--op denoise inpaint`: Forward operators for denoising and inpainting
- `--mask-path`: Path to the solvent mask (to speed up CryoFM inference)
- `--num_processes`: Number of GPUs to use for CryoFM2 processing
- `--external_reconstruct`: RELION flag to use external reconstruction executable

Adjust the number of processes, and other RELION parameters according to your dataset configuration.
