"""UKAST model definition.

The architecture deliberately delegates SwinUNETR-v2, including its residual
convolution stages and decoder, to MONAI.  UKAST changes only the transformer
feed-forward modules.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from monai.networks.nets import SwinUNETR

from .rational import GroupRationalKAN, TritonGroupRationalKAN


class UKAST(SwinUNETR):
    """SwinUNETR-v2 whose Swin FFNs are Group Rational KANs.

    The defaults reproduce the completed experiment family: four groups,
    (5,4) rational functions, identity then swish initialization, and v2
    residual convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        patch_size: int = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int | Sequence[int] = 7,
        feature_size: int = 24,
        mlp_ratio: float = 4.0,
        spatial_dims: int = 2,
        norm_name: str = "instance",
        group_size: int = 4,
        rational_init: str = "swish",
        backend: str = "triton",
        **swinunetr_kwargs: object,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            feature_size=feature_size,
            mlp_ratio=mlp_ratio,
            spatial_dims=spatial_dims,
            norm_name=norm_name,
            use_v2=True,
            **swinunetr_kwargs,
        )
        self.group_size = group_size
        self.rational_init = rational_init
        if backend not in {"triton", "torch"}:
            raise ValueError("backend must be 'triton' or 'torch'")
        self.backend = backend
        self._expected_ffns = sum(depths)
        self._replace_feed_forwards()

    def _replace_feed_forwards(self) -> None:
        replaced = 0
        for module in self.swinViT.modules():
            mlp = getattr(module, "mlp", None)
            if mlp is None or not hasattr(mlp, "linear1") or not hasattr(mlp, "linear2"):
                continue
            linear1, linear2 = mlp.linear1, mlp.linear2
            kan_type = TritonGroupRationalKAN if self.backend == "triton" else GroupRationalKAN
            module.mlp = kan_type(
                linear1.in_features,
                linear1.out_features,
                linear2.out_features,
                groups=self.group_size,
                activation_init=self.rational_init,
                dropout=0.0,
            )
            replaced += 1
        if replaced != self._expected_ffns:
            raise RuntimeError(f"Expected to replace {self._expected_ffns} Swin FFNs, replaced {replaced}.")
