# Common Issues

This page addresses frequently encountered issues and their solutions.

---

### Multi-GPU and Distributed Training

<div class="admonition info">
  <p class="admonition-title"><strong>Question</strong></p>
  <p>How do we launch an experiment with multiple GPUs or across multiple nodes?</p>
</div>

We run multi-GPU and distributed training using the [Lightning](https://lightning.ai/) framework. This section explains how our codebase is configured to work with distributed training.

**Configuration Flow**

Our distributed training setup works through the following components:

1. Launch with `torchrun`: We use `torchrun` to launch multi-node, multi-GPU training
2. Configuration via `--cfg-options`: Pass `pl_trainer.num_nodes` and `pl_trainer.devices` through `--cfg-options`
3. Config parsing: The `init_pl_w_cfg()` function in `src/cryofm/core/training/lightning_utils.py` uses `mmengine_parser()` to parse CLI arguments and merge them into the config via `cfg.merge_from_dict()`
4. Trainer initialization: The trainer is initialized with `**cfg.pl_trainer` to expand all Lightning Trainer parameters, along with `strategy=DDPStrategy(find_unused_parameters=True)`

**Example Command**

Here's how to launch distributed training with `torchrun`:

```bash
# Replace the variables with your actual cluster settings:
# NUM_NODES:           total number of nodes
# NODE_RANK:           rank of this node (starts from 0)
# NUM_GPUS_PER_NODE:   number of GPUs per node
# MASTER_ADDR:         master node IP or hostname
# MASTER_PORT:         port number for communication

torchrun --nnodes=${NUM_NODES} \
         --node_rank=${NODE_RANK} \
         --nproc_per_node=${NUM_GPUS_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         train.py configs/cryofm2/cryofm2_pretrain.py \
         --cfg-options pl_trainer.num_nodes=${NUM_NODES} pl_trainer.devices=${NUM_GPUS_PER_NODE}
```

**How It Works**

1. `torchrun` handles the process spawning and distributed communication setup
2. `--cfg-options` passes `pl_trainer.num_nodes` and `pl_trainer.devices` to override the config file defaults
3. `init_pl_w_cfg()` (in `src/cryofm/core/training/lightning_utils.py`) loads the config file and returns the merged `cfg` object
4. Trainer initialization: Your `train.py` should follow this pattern:

    ```python
    import lightning.pytorch as pl
    from lightning.pytorch.strategies import DDPStrategy

    cfg = init_pl_w_cfg(exp_prefix="cryofm", inplace=False)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        **cfg.pl_trainer  # Expands all pl_trainer config parameters
    )
    ```
    The `**cfg.pl_trainer` expansion includes `num_nodes` and `devices` that were passed via `--cfg-options`, which are then used by Lightning Trainer for correct distributed/multi-GPU setup.

---

### Converting Checkpoint to Safetensors Format

<div class="admonition info">
  <p class="admonition-title"><strong>Question</strong></p>
  <p>How do we convert training checkpoints to safetensors format?</p>
</div>

We use a conversion script to convert PyTorch Lightning checkpoint files to safetensors format. This script is designed to work with our custom EMA implementation (`cryofm.core.models.ema.LitEma`) and merges base model weights with EMA (Exponential Moving Average) weights.

**Conversion Script**

Here's the complete conversion script:

```python
from argparse import ArgumentParser
import torch
from safetensors.torch import save_file

DROP = {"decay", "num_updates"}

def parse_args():
    p = ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--ema_prefix", default="model_ema.")
    p.add_argument("--base_prefix", default="model.")
    return p.parse_args()

def strip_prefix(s: str, prefix: str):
    return s[len(prefix):] if prefix and s.startswith(prefix) else s

if __name__ == "__main__":
    args = parse_args()
    ckpt = torch.load(args.input_path, map_location="cpu")
    sd = ckpt["state_dict"]

    # 1) base: state_dict, including buffers
    base = {}
    for k, v in sd.items():
        if args.base_prefix == "":
            if not k.startswith(args.ema_prefix):
                base[k] = v
        else:
            if k.startswith(args.base_prefix):
                base[strip_prefix(k, args.base_prefix)] = v

    # 2) ema: our LitEma only include trainable params, and name should __ -> .
    ema = {}
    for k, v in sd.items():
        if k.startswith(args.ema_prefix):
            k = strip_prefix(k, args.ema_prefix)
            k = k.replace("__", ".")
            if k in DROP:
                continue
            ema[k] = v

    # 3) merge
    merged = dict(base)
    merged.update(ema)

    save_file(merged, args.output_path)
    print(f"Saved merged EMA weights: {len(merged)} tensors")
```

---
