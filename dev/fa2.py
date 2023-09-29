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
        strides = (stride_vk, stride_kn),
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


z,h,n_ctx,d_head = (1,2, 256, 16)
q = torch.randn((z,h,n_ctx, d_head), device='cuda')
k = torch.randn((z,h,n_ctx, d_head),device='cuda')
v = torch.randn((z,h,n_ctx, d_head),device='cuda')
causal=True
sm_scale = 0.5

res = attention(q,k,v, causal, sm_scale)
