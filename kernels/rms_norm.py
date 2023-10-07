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
def _fwd_rms_kernel(
    out_ptr_base,
    stride_out_row,
    in_ptr_base,
    stride_x_row,
    stride_x_col, 
    weight_ptr,
    rstd,
    num_rows: tl.constexpr,
    num_cols: tl.constexpr,
    block_size: tl.constexpr,

):
    # get input pointers ready
    row_index = tl.program_id(0)
    in_ptr_row = in_ptr_base + (row_index * stride_x_row)
    out_ptr_row = out_ptr_base + (row_index * stride_out_row)

    # Create block pointers for input - waiting for compiler bug fix
    #in_block_ptr = tl.make_block_ptr(base=in_ptr_base, shape=(num_rows, num_cols), strides=(stride_x_row, stride_x_col),
    #                                 offsets=(row_index, 0), block_shape=(1, block_size), order=(1, 0))
    

    # internal variables
    variance = 0.0
    eps=1e-8  # per RMSNorm official repo

    # rms_x = norm_x * d_x ** (-1. / 2)
    # x_normed = x / (rms_x + self.eps)
    
    for col_index in range(0, num_cols, block_size): 
        # col block 0.069104, 1.188494, -0.996045, 0.199789
        #col_block = tl.load(in_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        #variance += tl.sum(col_block * col_block, axis=0)
        #in_block_ptr = tl.advance(in_block_ptr, (0, block_size))

        col_offsets = col_index + tl.arange(0, block_size)
        col_mask = col_offsets < num_cols
        col_block = tl.load(in_ptr_row + col_offsets, mask = col_mask, other=0.0).to(tl.float32)
        
        variance += tl.sum(col_block * col_block, axis=0) 
        
    # complete the rms calcs
    variance /= num_cols
    rstdev = 1/ tl.sqrt(variance + eps)

    tl.store(rstd+row_index, rstdev )

    
    for start_col in range(0,num_cols, block_size):
        col_offsets = start_col + tl.arange(0, block_size)
        #print("col offsets", col_offsets)
        col_mask = col_offsets < num_cols
        weights = tl.load(weight_ptr + col_offsets, mask = col_mask)
        #in_block_ptr_offset = tl.advance(in_block_ptr, (0, start_col))
        #in_block = tl.load(in_block_ptr_offset, boundary_check=(0, 1), eviction_policy='evict_first')
        in_block = tl.load(in_ptr_row + col_offsets, mask=col_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        col_block_rms = in_block * rstdev
        out = weights * col_block_rms

        # write to HBM
        tl.store(out_ptr_row + col_offsets, out, mask = col_mask)
    

@triton.jit
def _rms_kernel_bwd_dx(
    dact,
    dout,
    in_ptr,
    stride_input_row,
    weight,
    nrows,
    ncols,
    block_size_cols: tl.constexpr,
    
): 
    row = tl.program_id(0)
    stride_to_row = row * stride_input_row
    input_row_ptr = in_ptr + stride_to_row

    dact_row_ptr = dact + stride_to_row
    dout_row_ptr = dout + stride_to_row


@triton.jit
def _rms_kernel_bwd_dwdb():
    pass

class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        weight: Tensor,
    ):
        
        # handle batches = flatten to 2D
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        
        nrows, ncols = x.shape


        out = torch.ones_like(x)
        rstd = torch.empty((nrows,), dtype=torch.float32, device="cuda")
        # block sizing - credit Kernl

        kb_64 = 65536 
        min_block_size = 128
        max_block_size = 4096
        default_num_warps = 8
        max_fused_size = kb_64 // x.element_size()
        block_size = min(max_fused_size, triton.next_power_of_2(ncols))
        block_size = max(block_size, min_block_size)
        block_size = min(block_size, max_block_size)

        print(f"Using block size {block_size=}")

        base_warps = max(block_size // 256, 1)
        num_warps = min(base_warps, 8)

        grid = (nrows,) # parallelize along rows
        _fwd_rms_kernel[grid](
            out_ptr_base=out,
            stride_out_row = out.stride(0),
            in_ptr_base = x,
            stride_x_row=x.stride(0),
            stride_x_col = x.stride(1),
            weight_ptr=weight,
            rstd=rstd,
            num_rows = nrows,
            num_cols = ncols,
            block_size=block_size,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x, weight, rstd)
        ctx.block_size = block_size
        ctx.num_warps = num_warps

        return out.view(*orig_shape)

    @staticmethod
    def backward(ctx, dout,):
        assert dout.is_contiguous()
        
        x, weight = ctx.saved_tensors
        dact = torch.empty_like(dout)
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        
        nrows, ncols = x.shape
        
        dweight = torch.empty((weight.shape[0],), dtype=weight.dtype, device=weight.device)
        grid = (nrows,)

        _rms_kernel_bwd_dx[grid](
            dact,
            dout,
            x,
            x.stride(0),
            weight,
            nrows,
            ncols,
            block_size_cols=ctx.block_size,
            num_warps=ctx.num_warps,
        )

        if ncols > 8192:   # kernl has 10240? 
            block_size_col = 128
            block_size_row = 32
            num_warps = 4
        else:
            block_size_col = 16
            block_size_row = 16
            num_warps = 8

        grid = lambda meta: [triton.cdiv(ncols, meta["block_size_col"])]

        _rms_kernel_bwd_dwdb[grid](
            x,
            dout,
            dweight,
            nrows,
            ncols,
            block_size_row=block_size_row,
            block_size_col=block_size_col,
            num_warps=num_warps,
        )
        
        return dact, dweight


# export function - allows typing of inputs
def triton_rmsnorm(
        x: Tensor,
        weight: Tensor,
        
):
    return TritonRMSNorm.apply(x, weight,)

