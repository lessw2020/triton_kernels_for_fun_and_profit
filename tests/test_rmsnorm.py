# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
import torch
import torch.nn as nn
from torch import Tensor
import sys
sys.path.append('..')
from kernels.rms_norm import triton_rmsnorm

from test_utils import assert_expected, set_rng_seed, gpu_test

@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(2020)

class TorchMM_RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    as proposed in: https://arxiv.org/abs/1910.07467

    Calcs are done in fp32.

    original impl: https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim(int) = model size
        eps(float) = epsilon
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        x_normed = self._norm(x.float()).type_as(x)
        return x_normed * self.scale
    

class TestRMSNorm:
    @pytest.fixture
    def N(self,):
        return 128
    
    #@gpu_test
    def test_triton_vs_pytorch_accuracy(self, N):
        batch_size = 512
        layer_weight_size = (N,N)
        layer_weight = torch.ones(layer_weight_size, requires_grad=False, device="cuda", dtype=torch.float32)

        sample_x = torch.randn(layer_weight_size, dtype=torch.float32, device='cuda', requires_grad=False)

        expected_rms_func = TorchMM_RMSNorm(layer_weight_size).to('cuda')
        expected_rms = expected_rms_func(sample_x)
        print(f"{expected_rms.shape=}")

        triton_out = triton_rmsnorm(sample_x, weight = layer_weight)
        print(f"{triton_out.shape=}")
        print(f"{triton_out=}")
        print(f"{expected_rms=}") # [:10,:10]

        assert_expected(triton_out, expected_rms)





        