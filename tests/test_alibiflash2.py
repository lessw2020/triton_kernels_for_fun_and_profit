# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
import torch
import torch.nn as nn
from torch import Tensor
import sys
sys.path.append('..')
from dev.fa2 import attention as alibi_attention
import time

from test_utils import assert_expected, set_rng_seed, gpu_test, perf_timer

@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(2023)


import math
from torch.nn.functional import scaled_dot_product_attention as sdpa

@perf_timer
def mha_compute(_func, q, k, v, causal, sm_scale, attn_mask = None, is_sdpa=False):
    
    if is_sdpa:
        res = _func(q,k,v, attn_mask=attn_mask, is_causal=causal, scale=sm_scale)
    else:
        res = _func(q,k,v, causal, sm_scale)
    return res


class TestAlibiFlash2:
    @pytest.fixture
    def N(self,):
        return 128 # 16384 # 8192
    
    @gpu_test()
    def test_forward(self, N):
        seq_len = N
        z,h,n_ctx,d_head = (1,1, seq_len, 16)
        print(f"in test")
        q = torch.randn((z,h,n_ctx, d_head), dtype=torch.bfloat16, device='cuda')
        k = torch.randn_like(q) # ((z,h,n_ctx, d_head),device='cuda')
        v = torch.randn_like(k) # ((z,h,n_ctx, d_head),device='cuda')
        torch.manual_seed(2020)
        q1 = torch.randn((z,h,n_ctx, d_head), dtype=torch.bfloat16, device='cuda')
        k1 = torch.randn_like(q) # ((z,h,n_ctx, d_head),device='cuda')
        v1 = torch.randn_like(k) # ((z,h,n_ctx, d_head),device='cuda')

        mask = torch.ones((seq_len, seq_len), dtype=torch.bfloat16, device='cuda')
        mask *=500

        torch.backends.cuda.enable_flash_sdp(False) # : Enables or Disables FlashAttention.

        torch.backends.cuda.enable_mem_efficient_sdp(True) # : Enables or Disables Memory-Efficient Attention.


        causal=True
        sm_scale = 1.0 # math.sqrt(k.shape[-1]) # 0.5
        # warmup
        triton_out, triton_time = mha_compute(alibi_attention, q,k,v, causal, sm_scale)
        # actual
        triton_out, triton_time = mha_compute(alibi_attention, q1,k1,v1, causal, sm_scale)
        print(f"{triton_out.dtype=}")
        
        use_manual = False
        if use_manual:
            M = torch.tril(torch.ones((n_ctx, n_ctx), device="cuda"))
            p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
            if causal:
                p[:, :, M == 0] = float("-inf")

            p = torch.softmax(p.float(), dim=-1)# .half()
            print(f"{p.dtype=}")
                # p = torch.exp(p)
            ref_out = torch.matmul(p.to(torch.bfloat16), v)
        # warmup
        causal=False
        sdpa_out, sdpa_time = mha_compute(sdpa, q,k,v, causal, sm_scale, mask, is_sdpa=True)
        # actual
        sdpa_out, sdpa_time = mha_compute(sdpa, q1,k1,v1, causal, sm_scale, mask, is_sdpa=True)
        print(f"{sdpa_out.dtype=}")
        print(f"timing compare: {triton_time=}, {sdpa_time=}")

        print(f"verifying output vs reference:")
        print(f"{sdpa_out[0][0][0][10:20]=}")
        print(f"{triton_out[0][0][0][10:20]=}")

        #distance_bias_matrix = -torch.abs(
        #            torch.arange(0,10) - torch.arange(0,10)[:,None]
            #       )
        #print(f"{distance_bias_matrix[0:10]=}")

        #torch.testing.assert_close(res, sdpa_out,atol=1e-1, rtol=0)
        #torch.testing.assert_close(ref_out, sdpa_out,atol=1e-1, rtol=0)

