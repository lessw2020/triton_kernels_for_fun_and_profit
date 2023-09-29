# mask support for Triton Flash2

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl

@triton.jit

def _attn_fwd_inner(
    accum, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    block_m: tl.constexpr,
    block_dmodel: tl.constexpr,
    block_n: tl.constexpr,
    stage: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    if stage==1:
        low= 0
        high = start_m * block_m
    else:
        low = start_m * block_m
        high = (start_m+1) * block_m
        low = tl.multiple_of(low, block_m) # compiler opt
    
    K_block_ptr = tl.advance(K_block_ptr, (0,low))
    V_block_ptr = tl.advance(V_block_ptr, (low,0))

    # loop KV and update accumulator
    for start_n in range(low, high, block_n):
        start_n = tl.multiple_of(start_n, block_n)
        # qk
        k = tl.load(K_block_ptr)
        qk = tl.zeros([block_m, block_n], dtype=tl.float32)
        qk += tl.dot(q,k)

        if stage==2:
            mask = offs_m[:,None] >= (start_n + offs_n[None,:])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk,axis=1))
            qk -= m_ij[:,None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk,axis=1) * qk_scale)
            qk = qk * qk_scale - m_ij[:,None]
        probs = tl.math.exp2(qk)
        l_ij = tl.sum(probs, axis=1)

        delta = tl.math.exp2(m_i - m_ij)
        l_i = l_i * delta + l_ij
        # update accum
        accum = accum * delta[:,None]
        v = tl.load(V_block_ptr)
        accum+= tl.dot(probs.to(v.type.element_ty), v)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (block_n,0))
        K_block_ptr = tl.advance(K_block_ptr, (0, block_n))
    return accum, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,K,V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    n_ctx: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_dmodel: tl.constexpr,
    stage: tl.constexpr,
):
    start_m = tl.program_id(0)
    
    off_hz = tl.program_id(1)
    off_z = off_hz //H  # div by batch
    off_h = off_hz % H # which head num
    # print("offsets: ",start_m, off_z, off_h)

    # adjust into qkv by moving along batch and head num
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    
    Q_block_ptr = tl.make_block_ptr(
        base = Q + qkv_offset,
        shape=(n_ctx, block_dmodel), # N, d
        strides = (stride_qm, stride_qk),
        offsets = (start_m * block_m, 0),
        block_shape = (block_m, block_dmodel), 
        order=(1,0),
    )

    K_block_ptr = tl.make_block_ptr(
        base = K + qkv_offset,
        shape = (block_dmodel, n_ctx), # d, N  (transposed)
        strides = (stride_kk, stride_kn),
        offsets = (0,0),
        block_shape = (block_dmodel, block_n),
        order = (0,1),
    )

    V_block_ptr = tl.make_block_ptr(
        base = V + qkv_offset,
        shape = (n_ctx, block_dmodel),
        strides = (stride_vk, stride_vn),
        offsets = (0,0),
        block_shape = (block_n, block_dmodel),
        order=(1,0),
    )

    O_block_ptr = tl.make_block_ptr(
        base = Out + qkv_offset,
        shape = (n_ctx, block_dmodel),
        strides =(stride_om, stride_on),
        offsets = (start_m * block_m,0),
        block_shape = (block_m, block_dmodel),
        order=(1,0),
    )

    # offsets
    offs_m = start_m * block_m + tl.arange(0,block_m)
    offs_n = tl.arange(0,block_n)

    # m and l
    m_i = tl.zeros([block_m], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_m], dtype=tl.float32) + 1.0
    accum = tl.zeros([block_m, block_dmodel], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504 # 1/log(2)

    # load q - stays in SRAM 
    q = tl.load(Q_block_ptr)
    # stage 1 - off-band (?)
    if stage & 1:
        accum, l_i, m_i = _attn_fwd_inner(
            accum, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, qk_scale, 
            block_m, block_dmodel, block_n, 
            1, offs_m, offs_n,
        )
    tl.debug_barrier()

    if stage & 2:
        accum, l_i, m_i = _attn_fwd_inner(
            accum, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            block_m, block_dmodel, block_n, 
            2, offs_m, offs_n,
        )

    # wrap up
    m_i += tl.math.log2(l_i)
    accum = accum / l_i[:,None]
    m_ptrs = M + off_hz * n_ctx + offs_m

    # back to SRAM
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, accum.to(Out.type.element_ty))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lk in {16,32,64,128}
        assert Lq == Lk and Lk==Lv

        out = torch.empty_like(q)

        block_m = 128
        block_n = 64 if Lk < 65 else 32 # 64
        num_stages = 4 if Lk < 65 else 3 # 64
        num_warps = 4

        grid_rows = (triton.cdiv(q.shape[2], block_m),)
        # b, nh, seq_len, hdim
        # 4, 12
        # example: 1024 seq_len / 128 = 8 blocks

        grid_cols = (q.shape[0] * q.shape[1],1) # 48, 1
        grid = grid_rows + grid_cols
        # (8,4,1)
        M = torch.empty(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float32)
        # M = 1,4,1024
        batch, num_heads, n_ctx, d_head = q.shape

        _attn_fwd[grid](
            q,k,v, sm_scale, M, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, num_heads,
            n_ctx = n_ctx,
            block_m = block_m,
            block_n = block_n,
            block_dmodel = Lk,
            stage=3,
            num_warps = num_warps,
            num_stages = num_stages,
        )

        ctx.save_for_backward(q, k, v, out, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return out


attention = _attention.apply

import time
def perf_timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output=func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start
        print(elapsed_time)
        return output, elapsed_time
    return wrapper

@perf_timer
def mha_compute(_func, q, k, v, causal, sm_scale, is_sdpa=False):
    
    if is_sdpa:
        res = _func(q,k,v, is_causal=causal, scale=sm_scale)
    else:
        res = _func(q,k,v, causal, sm_scale)
    return res

import math
from torch.nn.functional import scaled_dot_product_attention as sdpa

z,h,n_ctx,d_head = (8,32, 2048, 64)
q = torch.randn((z,h,n_ctx, d_head), dtype=torch.bfloat16, device='cuda')
k = torch.randn_like(q) # ((z,h,n_ctx, d_head),device='cuda')
v = torch.randn_like(k) # ((z,h,n_ctx, d_head),device='cuda')
torch.manual_seed(2020)
q1 = torch.randn((z,h,n_ctx, d_head), dtype=torch.bfloat16, device='cuda')
k1 = torch.randn_like(q) # ((z,h,n_ctx, d_head),device='cuda')
v1 = torch.randn_like(k) # ((z,h,n_ctx, d_head),device='cuda')

causal=True
sm_scale = 1.0 # math.sqrt(k.shape[-1]) # 0.5
# warmup
res, triton_time = mha_compute(attention, q,k,v, causal, sm_scale)
# actual
res, triton_time = mha_compute(attention, q1,k1,v1, causal, sm_scale)
print(f"{res.dtype=}")
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
sdpa_out, sdpa_time = mha_compute(sdpa, q,k,v, causal, sm_scale, is_sdpa=True)
# actual
sdpa_out, sdpa_time = mha_compute(sdpa, q1,k1,v1, causal, sm_scale, is_sdpa=True)
print(f"{sdpa_out.dtype=}")
print(f"timing compare: {triton_time=}, {sdpa_time=}")

print(f"verifying output vs reference:")
torch.testing.assert_close(res, sdpa_out,atol=1e-1, rtol=0)
#torch.testing.assert_close(ref_out, sdpa_out,atol=1e-1, rtol=0)


