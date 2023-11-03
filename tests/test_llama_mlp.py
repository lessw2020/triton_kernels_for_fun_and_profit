# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
import torch
import torch.nn as nn
from torch import Tensor
import sys
sys.path.append('..')
from kernels.llama_mlp import triton_llama_mlp

from test_utils import assert_expected, set_rng_seed, gpu_test
import triton

@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(2023)

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

def rms_pytorch(x: torch.Tensor, rms_wts: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms_wts

    
import time

class TestFusedMLP:
    @pytest.fixture
    def N(self,):
        return 4096 # 16384 # 8192
    @pytest.fixture
    def lin_layer(self,):
        return 11008
    
    #@gpu_test
    def test_triton_vs_pytorch_accuracy(self, N, lin_layer):
        x = torch.randn([1, 16, N], dtype=torch.float16, device="cuda")
        
        rms_wts = torch.randn([N], dtype=torch.float16, device="cuda") * 0.15
        lin1_w = torch.randn([lin_layer, N], dtype=torch.float16, device="cuda") * 0.15
        lin2_w = torch.randn([lin_layer, N], dtype=torch.float16, device="cuda") * 0.15


        def mlp_pytorch(x, lin1, lin2, rms_wts):
            
            x_norm_llama = rms_pytorch(x, rms_wts=rms_wts)
            a = torch.nn.functional.silu(torch.matmul(x_norm_llama, lin1_w.t()))
            b = torch.matmul(x_norm_llama, lin2_w.t())
            return a * b

        output_triton = triton_llama_mlp(x=x, lin1=lin1_w, lin2=lin2_w, rms_wts=rms_wts)
        print(f"{output_triton.shape=}")
        print(f"{output_triton[0][0][0:10]=}")

        ref_mlp_out = mlp_pytorch(x=x, lin1=lin1_w, lin2=lin2_w, rms_wts=rms_wts)
        print(f"{ref_mlp_out.shape=}")
        print(f"{ref_mlp_out[0][0][0:10]=}")

        assert torch.allclose(output_triton, ref_mlp_out, atol=1e-1), f"max diff: {torch.max(torch.abs(output_triton - ref_mlp_out))}"

        print("triton", triton.testing.do_bench(lambda: triton_llama_mlp(x=x, lin1=lin1_w, lin2=lin2_w, rms_wts=rms_wts), warmup=50))
        print("pytorch", triton.testing.do_bench(lambda: mlp_pytorch(x=x, lin1=lin1_w, lin2=lin2_w, rms_wts=rms_wts)))


        
