import pytest
import torch
from torch import Tensor

from quack.mhc_post import mhc_post, mhc_post_ref


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    device: str = "cuda",
) -> dict[str, Tensor]:
    torch.random.manual_seed(0)
    return {
        "x": torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device),
        "residual": torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device),
        "post_layer_mix": torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device),
        "comb_res_mix": torch.randn(
            (n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device
        ),
    }


@pytest.mark.parametrize("n0", [1, 2])
@pytest.mark.parametrize("n1", [4096])
@pytest.mark.parametrize("h", [1280, 2560, 7168])
def test_mhc_post_forward(n0: int, n1: int, h: int) -> None:
    test_data = _make_test_data(n0=n0, n1=n1, h=h, mhc_mult=4)

    out = mhc_post(**test_data)
    out_ref = mhc_post_ref(**test_data)

    assert out.shape == out_ref.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, out_ref)


@pytest.mark.parametrize("n1", [17, 18, 20])
def test_mhc_post_forward_uneven_tokens(n1: int) -> None:
    test_data = _make_test_data(n0=1, n1=n1, h=1280, mhc_mult=4)

    out = mhc_post(**test_data)
    out_ref = mhc_post_ref(**test_data)

    assert out.shape == out_ref.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, out_ref)
