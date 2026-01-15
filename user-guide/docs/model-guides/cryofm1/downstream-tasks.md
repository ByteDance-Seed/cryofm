# CryoFM1: Downstream Tasks

## Prerequisites

Before using CryoFM1, ensure you have:

#### 1. Install CryoFM with compatible dependencies

CryoFM1 uses the HDiT model architecture, which depends on the `natten` package. Different versions of `natten` have varying requirements for PyTorch and CUDA versions. For a reproducible installation, follow these steps:

```bash
# natten 0.17.5 uses type union syntax, you must use python >=3.10
conda create -n cryofm python=3.10 -y
conda activate cryofm

# Install PyTorch 2.5.1 with CUDA 12.4 support
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install natten 0.17.5 compatible with PyTorch 2.5.0 and CUDA 12.4
pip install natten==0.17.5+torch250cu124 -f https://whl.natten.org

# Clone and install CryoFM
git clone https://github.com/ByteDance-Seed/cryofm
cd cryofm
pip install .
```

#### 2. Download model checkpoints and configuration files

CryoFM1 model weights and configuration files are available for download from the [Hugging Face repository](https://huggingface.co/ByteDance-Seed/cryofm-v1). To download the model weights, first install the Hugging Face CLI tool:
```bash
pip install huggingface-hub
```
Then download all model files using:
```bash
huggingface-cli download ByteDance-Seed/cryofm-v1 --local-dir ./cryofm-v1
```
This will download all necessary model files (including `cryofm-s` and `cryofm-l`) to the `./cryofm-v1` directory. You can change `./cryofm-v1` to your preferred download location.

---

## Prepare our test set

To prepare the test dataset for downstream tasks, you need to:

#### 1. Download the data list from Zenodo

Download the CSV file containing EMDB IDs from the [Zenodo repository](https://zenodo.org/records/18013604). The CSV file should have an `id` column with EMDB entry IDs (e.g., `EMD-12345`).

#### 2. Download and preprocess EMDB data

Use the `scripts/prepare_cryofm1_dataset.py` script to download density maps from EMDB and preprocess them:

```bash
python scripts/prepare_cryofm1_dataset.py \
    --data-list-path ./cryoFM-emdb-lists/cryofm1/cryofm1_1-5apix_dataset/test.csv \
    --save-dir ./cryofm1_1-5apix_dataset
```

**Parameters:**

- `--data-list-path`: Path to the CSV file downloaded from Zenodo containing EMDB IDs
- `--save-dir`: Directory where downloaded and processed maps will be saved

The script will:

- Download density maps from EMDB for each entry in the CSV file
- Extract the compressed map files
- Rescale all maps to 1.5 Angstrom/pixel resolution
- Create the required directory structure with a `split/` subdirectory containing the CSV file

**Note:** The script requires `relion_image_handler`, which is a command from the RELION software package. Please install RELION and ensure `relion_image_handler` is available in your PATH for rescaling operations. For installation instructions, refer to the [RELION Installation Guide](https://relion.readthedocs.io/en/release-5.0/Installation.html).

You will get a test dataset with the following structure:

```
path_to/cryofm1_1-5apix_dataset/
├── split/
│   └── test.csv
├── EMD-12042
└── ...
```

---

## Synthetic Downstream Tasks

### What is an Inverse Problem?

In cryo-EM, an inverse problem refers to the task of recovering the original, clean 3D density map from observed data that has been degraded by various factors during the imaging and reconstruction process. These degradations can include noise, missing information (such as the missing wedge in cryo-ET), and other artifacts. Solving inverse problems requires understanding the forward degradation process and then inverting it to restore the original signal.

CryoFM1 can be used to solve various inverse problems in cryo-EM through Flow Posterior Sampling. By leveraging the learned prior distribution of clean density maps, CryoFM1 can effectively restore degraded observations back to high-quality 3D structures. We have open-sourced implementations for three representative inverse problem tasks: **denoising**, **anisotropy denoising**, and **missing wedge restoration**.

---

### Run Downstream Tasks

#### Script Description

The `scripts/test_cryofm1.py` script performs downstream tasks using Flow Posterior Sampling. It processes each test sample by:

1. Applying a forward degradation operator (noise, missing wedge, etc.)
2. Generating two degraded half-maps
3. Estimating degradation operators from the half-map FSC
4. Running DPS to restore the clean density map
5. Computing FSC metrics between restored and ground truth maps

**Common Parameters:**

- `--data-root`: Path to your test dataset directory
- `--model-dir`: Path to the model directory (e.g., `path_to/cryofm-v1/cryofm-s/`)
- `--bf16`: Enable bfloat16 mixed precision inference to significantly speed up computation. Requires GPU hardware support for `bfloat16` (e.g., NVIDIA A100, H100, or newer architectures).
- `--exp-name`: Experiment name for organizing output files
- `--num-timesteps`: Number of sampling steps (use 1000 to reproduce the best results reported in our paper)
- `--task-names`: Task type(s) to perform
- `--eval-n-samples`: Number of samples to evaluate (default: all samples in test set)

**Output:**

Results are saved in `work_dirs/{exp_name}/{map_id}/` for each test sample, including:

- Restored density maps (`new_y1.mrc`, `new_y2.mrc`, `new_y_avg.mrc`)
- FSC curves comparing restored maps with ground truth
- Statistics and metrics in the log file

**Note:**

- Processing 32 test cases typically takes approximately 4 hours for denoising and anisotropic denoising tasks, and approximately 1 hour for missing wedge restoration, on a single NVIDIA A100 GPU with `bfloat16` mixed precision enabled, using `1000` sampling steps.
- You can adjust posterior sampling hyper-parameters with `--lamb-base` and `--lamb-w-max` if needed.
- Check the FSC curves in the output directory to evaluate restoration quality.

---

#### Spectral Noise Denoising

Removes isotropic spectral noise from cryo-EM density maps. The noise power spectrum is estimated from FSC curves based on the specified SNR index.

```bash
python scripts/test_cryofm1.py \
    --data-root path_to/cryofm1_1-5apix_dataset/ \
    --model-dir path_to/cryofm-v1/cryofm-s/ \
    --exp-name cryofm_sn_snr1 \
    --num-timesteps 1000 \
    --task-names spectral_noise \
    --snr-idx 1 \
    --bf16
```

**Parameters:**

- `--snr-idx`: SNR index (1-5) corresponding to different FSC-based SNR levels. Higher indices indicate higher noise levels.

**Results:**

After the script completes, it will print statistical results for all 32 test cases, including detailed results for each case and summary statistics (mean and standard deviation). 

Example summary output:

```text
Summary Statistics:
01/05 17:21:45 - __main__ - INFO - 
            type  mean  std
0    observation  0.xx 0.0x
1          pred1  0.xx 0.0x
2          pred2  0.xx 0.0x
3  pred half avg  0.xx 0.0x
```

The following table shows the $\mathrm{FSC}_\mathrm{AUC}$ results reported in our paper for reference:

| Type | SNR index 1 | SNR index 2 | SNR index 3 | SNR index 4 | SNR index 5 |
|------|----------|----------|----------|----------|----------|
| observation | 0.8876 | 0.5527 | 0.4969 | 0.3275 | 0.2014 |
| prediction half avg. | 0.95 ± 0.01 | 0.68 ± 0.04 | 0.62 ± 0.04 | 0.38 ± 0.01 | 0.24 ± 0.01 |

---

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
    --tilt-angle 15 \
    --bf16
```

**Parameters:**

- `--tilt-angle`: Maximum tilt angle in degrees (e.g., 15, 30, 45). The code simulates anisotropy by amplifying noise levels outside the tilt angle region. Smaller angles result in larger regions with amplified noise, making the task more challenging.

**Results:**

The following table shows the $\mathrm{FSC}_\mathrm{AUC}$ results reported in our paper for reference:

| Type | tilt_angle 45 | tilt_angle 30 | tilt_angle 15 |
|------|----------|----------|----------|
| observation | 0.6623 | 0.6324 | 0.6111 |
| prediction half avg. | 0.88 ± 0.03 | 0.84 ± 0.03 | 0.81 ± 0.04 |

---

#### Missing Wedge Restoration

Restores information lost due to the missing wedge in cryo-ET reconstructions. The missing wedge is defined by the maximum tilt angle.

```bash
python scripts/test_cryofm1.py \
    --data-root path_to/cryofm1_1-5apix_dataset/ \
    --model-dir path_to/cryofm-v1/cryofm-s/ \
    --exp-name cryofm_mw_tilt60 \
    --num-timesteps 1000 \
    --task-names missing_wedge \
    --tilt-angle 60 \
    --bf16
```

**Parameters:**

- `--tilt-angle`: Maximum tilt angle in degrees. The missing wedge corresponds to the unobserved region in Fourier space beyond this angle.

**Results:**

The following table shows the $\mathrm{FSC}_\mathrm{AUC}$ results reported in our paper for reference:

| Type | tilt_angle 60 |
|------|----------|
| observation | 0.80 ± 0.02 |
| prediction half avg. | 0.92 ± 0.02 |
