from itertools import product

import pytest
import torch
from torch import Tensor
from pyg_lib.ops.fps import fps
from pyg_lib.testing import devices, grad_dtypes, tensor


@torch.jit.script
def fps2(x: Tensor, ratio: Tensor) -> Tensor:
    return fps(x, None, ratio, False)


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_fps(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-2, -2],
        [-2, +2],
        [+2, +2],
        [+2, -2],
    ], dtype, device)
    batch = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)

    out = fps(x, batch, random_start=False)
    assert out.tolist() == [0, 2, 4, 6]

    out = fps(x, batch, ratio=0.5, random_start=False)
    assert out.tolist() == [0, 2, 4, 6]

    out = fps(x, batch, ratio=torch.tensor(0.5, device=device),
              random_start=False)
    assert out.tolist() == [0, 2, 4, 6]

    out = fps(x, batch, ratio=torch.tensor([0.5, 0.5], device=device),
              random_start=False)
    assert out.tolist() == [0, 2, 4, 6]

    out = fps(x, random_start=False)
    assert out.sort()[0].tolist() == [0, 5, 6, 7]

    out = fps(x, ratio=0.5, random_start=False)
    assert out.sort()[0].tolist() == [0, 5, 6, 7]

    out = fps(x, ratio=torch.tensor(0.5, device=device), random_start=False)
    assert out.sort()[0].tolist() == [0, 5, 6, 7]

    out = fps(x, ratio=torch.tensor([0.5], device=device), random_start=False)
    assert out.sort()[0].tolist() == [0, 5, 6, 7]

    out = fps2(x, torch.tensor([0.5], device=device))
    assert out.sort()[0].tolist() == [0, 5, 6, 7]


@pytest.mark.parametrize('device', devices)
def test_random_fps(device):
    N = 1024
    for _ in range(5):
        pos = torch.randn((2 * N, 3), device=device)
        batch_1 = torch.zeros(N, dtype=torch.long, device=device)
        batch_2 = torch.ones(N, dtype=torch.long, device=device)
        batch = torch.cat([batch_1, batch_2])
        idx = fps(pos, batch, ratio=0.5)
        assert idx.min() >= 0 and idx.max() < 2 * N
