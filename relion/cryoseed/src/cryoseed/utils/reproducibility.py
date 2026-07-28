import hashlib
import os
import random

import numpy as np
import torch


def derive_seed(seed: int, *items) -> int:
    """Derive a stable 63-bit seed from a base seed and context items.

    Tensor inputs are hashed from their int64 CPU values in order, while
    non-tensor items are hashed from their ``repr``. The returned seed is
    suitable for ``torch.Generator.manual_seed``.

    This helper is intended for integer tensors, scalar values, and string
    labels. It should not be treated as a general-purpose hash for arbitrary
    floating-point tensors or complex Python objects.

    The derived seed is useful for batch-level reproducibility, e.g. to give
    one batched random draw a stable context-dependent seed. It does not imply
    that individual samples remain invariant when batch composition or ordering
    changes.
    """
    max_seed = (1 << 63) - 1
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update((int(seed) % max_seed).to_bytes(8, "little", signed=False))

    for item in items:
        if torch.is_tensor(item):
            tensor = (
                item.detach()
                .to(device="cpu", dtype=torch.int64)
                .reshape(-1)
                .contiguous()
            )
            hasher.update(b"tensor")
            hasher.update(int(tensor.numel()).to_bytes(8, "little", signed=False))
            for value in tensor.tolist():
                hasher.update(int(value).to_bytes(8, "little", signed=True))
        else:
            hasher.update(b"repr")
            hasher.update(repr(item).encode("utf-8"))

    return int.from_bytes(hasher.digest()[:8], "little", signed=False) % max_seed


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): base seed
        deterministic (bool): enforce deterministic cudnn behavior
    """

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Hash seed (for dataloader workers etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False