# CryoFM2: Unconditional Sampling

## Prerequisites

Before using CryoFM2, please ensure:

1. **Install CryoFM**: Follow the [Installation Guide](../../getting-started/installation.md) to install `cryofm`.
2. **Download Model Weights**: CryoFM2 model weights are available for download from [Hugging Face](https://huggingface.co/ByteDance-Seed/cryofm-v2). To download the model weights, first install the Hugging Face CLI tool:
   ```bash
   pip install huggingface_hub
   ```
   Then download all model files using:
   ```bash
   hf download ByteDance-Seed/cryofm-v2 --local-dir ./cryofm-v2
   ```
   This will download all necessary model files (including `cryofm2-pretrain`, `cryofm2-emhancer`, and `cryofm2-emready`) to the `./cryofm-v2` directory. You can change `./cryofm-v2` to your preferred download location.

## Generating Samples
> Exploring Training Data Distribution

Generate samples from the pretrained model to explore the learned data distribution. This is useful for understanding what the model has learned and for generating synthetic density maps.

<figure markdown>
  ![CryoFM2 samples](../../images/cryofm2_uncond-examples.jpg)
  <figcaption>CryoFM2 unconditional samples.</figcaption>
</figure>

#### Pretrained Model

```python
import torch
from mmengine import Config

from cryofm.core.utils.mrc_io import save_mrc
from cryofm.core.utils.sampling_fm import sample_from_fm
from cryofm.projects.cryofm2.lit_modules import CryoFM2Uncond

# Load configuration and model
# Update the path to your model directory
model_dir = "path/to/cryofm-v2/cryofm2-pretrain"
cfg = Config.fromfile(f"{model_dir}/config.yaml")
lit_model = CryoFM2Uncond.load_from_safetensors(
    f"{model_dir}/model.safetensors", 
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
        device=lit_model.device, 
        side_shape=64
    )

# Save generated density maps
for i in range(3):
    save_mrc(
        out[i].float().cpu().numpy(), 
        f"sample-{i}.mrc", 
        apix=1.5  # Angstroms per pixel
    )
```

#### Fine-tuned Models (EMhancer/EMReady)

Fine-tuned models can also generate unconditional samples in their respective styles:

```python
import torch
from mmengine import Config

from cryofm.core.utils.mrc_io import save_mrc
from cryofm.core.utils.sampling_fm import sample_from_fm
from cryofm.projects.cryofm2.lit_modules import CryoFM2Cond

# Choose style: "emhancer" or "emready"
style = "emhancer"
model_dir = f"path/to/cryofm-v2/cryofm2-{style}"
cfg = Config.fromfile(f"{model_dir}/config.yaml")
lit_model = CryoFM2Cond.load_from_safetensors(
    f"{model_dir}/model.safetensors", 
    cfg=cfg
)
output_tag = 1 if style == "emhancer" else 0

# Set up device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_model = lit_model.to(device)
lit_model.eval()

# Define vector field function with conditional generation
def v_xt_t(_xt, _t):
    bs = _xt.shape[0]
    unconditional_generation_conds = {
        "input_cond": None,
        "output_cond": torch.tensor([output_tag] * bs).to(device),
        "vol_cond": None,  # dimension should be [bs, d, h, w]
    }
    return lit_model(_xt, _t, generation_conds=unconditional_generation_conds)

# Generate samples
# Note: Enable bfloat16 if your GPU supports it for better performance
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = sample_from_fm(
        v_xt_t, 
        lit_model.noise_scheduler, 
        method="euler", 
        num_steps=200, 
        num_samples=3, 
        device=lit_model.device, 
        side_shape=64
    )

# Save generated density maps
for i in range(3):
    save_mrc(
        out[i].float().cpu().numpy(), 
        f"{style}-sample-{i}.mrc", 
        apix=1.5  # Angstroms per pixel
    )
```

