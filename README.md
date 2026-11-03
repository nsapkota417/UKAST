# UKAST

Reference implementation for **When Swin Transformer Meets KANs: An Improved
Transformer Architecture for Medical Image Segmentation** (ISBI 2026).

Read the [paper](https://arxiv.org/pdf/2511.04084).

UKAST retains MONAI's `SwinUNETR` decoder and **v2 residual-convolution
encoder stages**. Its sole architectural change is replacing every Swin MLP
with KAT's token-wise Group Rational KAN:

```text
Group Rational(identity) -> Linear -> Group Rational(swish) -> Linear
```

The public implementation intentionally excludes register tokens, experimental
architectures, dataset copies, cluster scripts, and checkpoints.

## Reproducible model configuration

The canonical configuration is in `configs/ukast_isbi2026.yaml`:

- 2D slice-wise segmentation, feature size 24, depths `(2, 2, 2, 2)`
- `group_size=4`, rational order fixed to `(5, 4)`, swish initialization
- MONAI SwinUNETR-v2 residual convolutions enabled
- AdamW, learning rate `5e-4`, weight decay `1e-3`, 400 epochs

`(5,4)` is fixed deliberately. The KAT Triton kernel used in the experiments
is hard-coded for a fifth-degree numerator and fourth-degree denominator;
configuring other orders is not supported by the release.

## Install

```bash
pip install -e '.[triton,dev]'
```

Install a CUDA-compatible PyTorch build for GPU training. The default model
uses the same 1D `KAT_Group` Triton module selected by the completed
experiments. For CPU-only inspection or unit tests, pass `backend="torch"`;
that selects the included mathematical reference implementation.

## Model

```python
from ukast import UKAST

model = UKAST(
    in_channels=3,
    out_channels=2,
    spatial_dims=2,
    group_size=4,
    rational_init="swish",
    backend="triton",
)
```

Inputs must satisfy MONAI SwinUNETR's spatial divisibility constraint (32 per
dimension for `patch_size=2`).

## Attribution

UKAST builds on [MONAI SwinUNETR](https://github.com/Project-MONAI/MONAI)
(Apache-2.0) and the [KAT rational activation implementation](https://github.com/Adamdad/rational_kat_cu).
See their licenses before redistributing modified upstream code or kernels.
