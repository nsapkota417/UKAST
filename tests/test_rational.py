import pytest

torch = pytest.importorskip("torch")

from ukast.rational import GroupRational, GroupRationalKAN


def test_identity_initialization_is_identity() -> None:
    layer = GroupRational(groups=4, init="identity")
    x = torch.randn(2, 7, 24)
    assert torch.allclose(layer(x), x)


def test_grkan_preserves_token_shape() -> None:
    layer = GroupRationalKAN(24, 96, 24, groups=4)
    assert layer(torch.randn(2, 11, 24)).shape == (2, 11, 24)
