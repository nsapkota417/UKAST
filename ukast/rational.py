"""The 1D Group Rational KAN used by the published UKAST experiments.

This is a small, framework-native reference implementation of the KAT
``KAT_Group`` operation.  It intentionally supports only the effective
experiment configuration: a fifth-degree numerator and fourth-degree
denominator.  The original Triton kernel is hard-coded to those degrees too;
pretending other orders work would make a non-reproducible release.
"""

from __future__ import annotations

import json
from importlib.resources import files

import torch
from torch import nn


def _initial_coefficients(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the KAT (5, 4) initialization coefficients."""
    with files("ukast").joinpath("kat_init.json").open() as stream:
        values = json.load(stream)
    if mode not in values:
        valid = ", ".join(sorted(values))
        raise ValueError(f"Unknown rational initialization {mode!r}; choose one of: {valid}.")
    config = values[mode]
    return (
        torch.tensor(config["init_w_numerator"], dtype=torch.float32).view(1, 6),
        torch.tensor(config["init_w_denominator"], dtype=torch.float32),
    )


class GroupRational(nn.Module):
    """KAT's group-wise rational activation for channels-last token tensors.

    Each channel group uses one denominator polynomial; the numerator is shared
    by all groups, exactly as in the vendored KAT implementation used in the
    experiments.  Input must have shape ``(..., channels)``.
    """

    order = (5, 4)

    def __init__(self, groups: int = 4, init: str = "swish") -> None:
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be positive")
        numerator, denominator = _initial_coefficients(init)
        self.groups = groups
        self.numerator = nn.Parameter(numerator)
        self.denominator = nn.Parameter(denominator.repeat(groups, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("expected a tensor with a channel dimension")
        channels = x.shape[-1]
        if channels % self.groups:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({self.groups})")

        shape = (*x.shape[:-1], self.groups, channels // self.groups)
        z = x.reshape(shape)
        powers = torch.stack([z.pow(i) for i in range(6)], dim=-1)
        coefficient_shape = (1,) * (z.ndim - 2) + (self.groups, 1, 6)
        numerator = torch.sum(powers * self.numerator.view(coefficient_shape), dim=-1)
        absolute_powers = torch.stack([z.abs().pow(i) for i in range(1, 5)], dim=-1)
        denominator_shape = (1,) * (z.ndim - 2) + (self.groups, 1, 4)
        denominator = 1.0 + torch.sum(
            absolute_powers * self.denominator.abs().view(denominator_shape), dim=-1
        )
        return (numerator / denominator).reshape_as(x)


class GroupRationalKAN(nn.Module):
    """The KAT feed-forward layout: GR Act -> Linear -> GR Act -> Linear."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        groups: int = 4,
        activation_init: str = "swish",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.act1 = GroupRational(groups=groups, init="identity")
        self.drop1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.act2 = GroupRational(groups=groups, init=activation_init)
        self.drop2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.drop2(self.act2(self.linear1(self.drop1(self.act1(x))))))


class TritonGroupRationalKAN(nn.Module):
    """Exact KAT Triton FFN used for the completed experiments.

    ``kat_rational`` is intentionally an optional dependency because its
    Triton extension requires a CUDA build matching the installed PyTorch.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        groups: int = 4,
        activation_init: str = "swish",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from kat_rational import KAT_Group
        except ImportError as error:
            raise ImportError(
                "The published Triton path requires `pip install -e '.[triton]'`. "
                "Use backend='torch' only for CPU/reference execution."
            ) from error
        self.act1 = KAT_Group(num_groups=groups, mode="identity", device="cuda")
        self.drop1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.act2 = KAT_Group(num_groups=groups, mode=activation_init, device="cuda")
        self.drop2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.drop2(self.act2(self.linear1(self.drop1(self.act1(x))))))
