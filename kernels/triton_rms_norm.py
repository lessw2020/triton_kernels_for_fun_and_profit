# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# ---- Fused RMSNorm written in Triton ------
# Extra Credits:
# Kernl LayerNorm in Triton

import torch
import triton
import triton.language as tl
from torch import Tensor  # typing
from typing import Optional
from torch.autograd.function import FunctionCtx


@triton.jit
def _fwd_rms_kernel():
    pass

class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor]

    ):
        
        # handle batches = flatten to 2D
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        nrows, ncols = x.shape

        out = torch.empty_like(x)

        # block sizing

        kb_64 = 65536 
        min_block_size = 128
        max_block_size = 4096
        default_num_warps = 8
        max_fused_size = kb_64 // x.element_size()
        block_size = min(max_fused_size, triton.next_power_of_2(ncols))
        block_size = max(block_size, min_block_size)
        block_size = min(block_size, max_block_size)

        base_warps = max(block_size // 256, 1)
        num_warps = min(base_warps, 8)

        grid = (nrows,) # parallelize along rows
        _fwd_rms_kernel[grid](
            out_ptr=out,
            in_ptr = x,
            stride_x_row=x.stride(0),
            weight=weight,
            num_cols = ncols,
            block_size=block_size,
            num_warps=num_warps,
        )







# export function - allows typing of inputs
def triton_rmsnorm(
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
):
    return TritonRMSNorm.apply(x, weight, bias)

