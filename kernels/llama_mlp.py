# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# ---- Fused LLama-MLP (silu) written in Triton ------
# Extra Credits:
# Kernl LayerNorm in Triton

import torch
import triton
import triton.language as tl
from triton.language import cdiv
from torch import Tensor  # typing
from typing import Optional
from torch.autograd.function import FunctionCtx
from math import ceil


@triton.jit
def _mlp_llama(act_ptr, lin1_ptr, lin2_ptr, out_ptr, rms_wt_ptr, 
               M, N, K,
               stride_act_m, stride_act_k,
               stride_lin1k, stride_lin1n,
               stride_lin2k, stride_lin2n,
               stride_outm, stride_outn,
               stride_rms_wt,
               eps: tl.constexpr, 
               block_size_m: tl.constexpr,
               block_size_n: tl.constexpr,
               block_size_k: tl.constexpr,):
    """ fwd impl for llama mlp """
    pid = tl.program_id(axis=0)
    step_n = cdiv(N, block_size_n)
    pid_m = pid // step_n
    pid_n = pid % step_n


    offs_k = tl.arange(0, block_size_k)
    offs_am = (pid_m * block_size_m + tl.arange(0,block_size_m)) % M
    offs_bn = (pid_n * block_size_n + tl.arange(0, block_size_n)) % N

    # zeros_block_size
    #zero_block_size = (block_size_m, block_size_n)

    act_ptrs = act_ptr + (offs_am[:, None]* stride_act_m + offs_k[None,:] * stride_act_k)
    lin1_ptrs = lin1_ptr + (offs_k[:,None] * stride_lin1k + offs_bn[None, :] * stride_lin1n)
    lin2_ptrs = lin2_ptr + (offs_k[:,None] * stride_lin2k + offs_bn[None,:] * stride_lin2n)

    acc1 = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    acc2 = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)

    rms_wt_ptrs = rms_wt_ptr + tl.arange(0, block_size_k)[None,:] * stride_rms_wt

    act_sum = tl.zeros((block_size_m, block_size_k), dtype = tl.float32)
    for i in range(0, cdiv(K, block_size_k)):
        act = tl.load(act_ptrs)
        act_sum += tl.math.pow(act.to(tl.float32), 2)
        rms_wt = tl.load(rms_wt_ptrs)
        act = act * rms_wt
        b = tl.load(lin1_ptrs)

        acc1 += tl.dot(act, b)
        c = tl.load(lin2_ptrs)
        acc2 += tl.dot(act, c)

        # advance (TODO: block ptrs)
        act_ptrs += block_size_k * stride_act_k
        lin1_ptrs += block_size_k * stride_lin1k
        lin2_ptrs += block_size_k * stride_lin2k
        rms_wt_ptrs += block_size_k * stride_rms_wt

    
    #rms norm
    act_mean = tl.sum(act_sum, axis=1) / K +eps
    act_norm = tl.math.rsqrt(act_mean)
    acc1 = acc1 * act_norm[:, None]
    acc2 = acc2 * act_norm[:, None]
    accum = (acc1 * tl.sigmoid(acc1) * acc2)

    offs_outm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_outn = pid_n * block_size_n + tl.arange(0,block_size_n)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:,None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:,None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accum, mask = out_mask)


def triton_llama_mlp(x: torch.Tensor, lin1: torch.Tensor, lin2: torch.Tensor, rms_wts: torch.Tensor):
    """ wrapper func for triton fused llama mlp """
    lin1_t = lin1.t()
    lin2_t = lin2.t()
    batch, seq_len, edim = x.shape

    M, K, = batch * seq_len, edim
    N = lin1_t.shape[1]
    x_reshape = x.reshape(M, K)
    out = torch.empty((M, N), dtype = x.dtype, device=x.device)


    # grid = (ceil(M // block_size_m) * ceil(N // block_size_n))
    grid = lambda meta: (triton.cdiv(meta["M"], meta["block_size_m"]) * triton.cdiv(meta["N"], meta["block_size_n"]),)
    
    kres = _mlp_llama[grid](x_reshape, lin1_t, lin2_t, out, rms_wts, M, N, K, 
                            *x_reshape.stride(), 
                            *lin1_t.stride(),
                            *lin2_t.stride(),
                            *out.stride(),
                            *rms_wts.stride(),
                            eps=1e-6,
                            block_size_m=16,
                            block_size_n=16,
                            block_size_k=64,
                            num_stages = 2,
                            num_warps=4,)
    out = out.view(batch, seq_len, -1)
    return out
   
    


    
    
    



