import triton
import triton.language as tl
import torch
from triton.runtime.driver import CudaUtils

#from base_gptq import triton_matmul4
import sys
sys.path.append('..')
from base_gptq import triton_matmul4 as no_autotune
# from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
from base_gptq import QuantLinear
#from gptq import small_qlinear # as sqlinear

@triton.jit()
def _small_quantized_matmul(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                             stride_am, stride_ak,
                             stride_bk, stride_bn,
                             stride_cm, stride_cn,
                             stride_scales_g, stride_scales_n,
                             stride_zeros_g, stride_zeros_n,
                             groupsize,
                             m, n, k,
                             block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr,
                             group_size_m: tl.constexpr,
                             fp8_fast_accum: tl.constexpr,):

    pid = tl.program_id(0)

    total_blocks_m = tl.cdiv(m, block_size_m)
    total_blocks_n = tl.cdiv(n, block_size_n)
    total_blocks_k = tl.cdiv(k, block_size_k)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_m = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_n = pid_n * block_size_n + tl.arange(0, block_size_n)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_size_m), block_size_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_size_n), block_size_n)

    offs_k = tl.arange(0, block_size_k)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) # (16, 64)
    # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(m,k), strides=(stride_am, stride_ak),
    #                                 offsets=(pid_m*block_size_m, 0), block_shape=(block_size_m, block_size_k),
    #                                 order =(1,0))

    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4


    output = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):

        # a = tl.load(a_block_ptr, boundary_check=(0,1))
        a = tl.load(a_ptrs)
        # print(a.type)
        b = tl.load(b_ptrs)
        g_id = k // (groupsize // block_size_k)

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)


        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]
        # output +=  tl.dot(a, b)
        # output += tl.sum(a, b, axis=0)
        # print(b.type)
        # result = a[:, None] * b # (1 x 64 x 64 x 32) x illegal # (NEED A SQUARE MATRIX for B)
        # b -> 64 x 64 instead 64 x 32

        output += tl.dot(a, b)
        # result = a[:, None] * b
        # output += tl.sum(result, 2)

        # a_block_ptr = tl.advance(a_block_ptr, (0, block_size_k))
        a_ptrs += stride_ak * block_size_k
        b_ptrs +=  (block_size_k//8) * stride_bk

    output.to(tl.float16)
    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    tl.store(c_ptrs, output)

    # offs_cn = pid_m * block_size_n + tl.arange(0, block_size_m)

    # tl.store(a_ptrs, a)
    # offs_k
    # offs_m


    # acc.to(tl.float16)
    # offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < n) & (offs_cn[None, :] < n)
    # tl.store(c_ptrs, acc, mask=c_mask)



class small_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros):

        m, k = a.shape
        _, n = b.shape

        quant_groupsize = 128
        block_size_m = 16
        block_size_n = 32 # [N = 4096 // 32] = 128 blocks
        block_size_k = 256 # 256 # [total_blocks_k = 4096 // 4096]
        group_size_m = 8
        num_warps = 4
        num_stages = 4
        total_blocks_m = triton.cdiv(m, block_size_m)
        total_blocks_n = triton.cdiv(n, block_size_n)
        total_programs  = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)
        fp8_fast_accum = False

        c = torch.zeros((m, n), device=b.device, dtype=torch.float16)
        # output = torch.em
        k = _small_quantized_matmul[grid](
            a, b, c, scales, zeros,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            quant_groupsize,
            m, n, k,
            block_size_m, block_size_n, block_size_k, group_size_m, fp8_fast_accum = fp8_fast_accum,
            num_warps = num_warps, num_stages = num_stages,
        )

        print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c


small_qlinear = small_qlinear.apply


@triton.jit()
def _base_quantized_matmul(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                             stride_am, stride_ak,
                             stride_bk, stride_bn,
                             stride_cm, stride_cn,
                             stride_scales_g, stride_scales_n,
                             stride_zeros_g, stride_zeros_n,
                             groupsize,
                             m, n, k,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, block_size_k: tl.constexpr,
                             group_size_m: tl.constexpr,
                             fp8_fast_accum: tl.constexpr,):

    pid = tl.program_id(0)

    total_blocks_m = tl.cdiv(m, BLOCK_M)
    total_blocks_n = tl.cdiv(n, BLOCK_N)
    total_blocks_k = tl.cdiv(k, block_size_k)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    offs_k = tl.arange(0, block_size_k)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < m)

    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, total_blocks_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)
        g_id = k // (groupsize // block_size_k)

        ptr = scales_ptrs + g_id * stride_scales_g

        scales = tl.load(ptr)
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        if fp8_fast_accum:
            acc = tl.dot(a.to(tl.float8e4nv), b.to(tl.float8e4nv), acc)
        else:
            acc += tl.dot(a,b)

        a_ptrs += block_size_k * stride_ak
        b_ptrs += (block_size_k//8) * stride_bk

    acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < n) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, acc, mask=c_mask)





class base_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros):

        m, k = a.shape
        _, n = b.shape

        quant_groupsize = 128
        BLOCK_M = 16
        BLOCK_N = 32
        block_size_k = 64
        group_size_m = 8
        num_warps = 4
        num_stages = 4
        total_blocks_m = triton.cdiv(m, BLOCK_M)
        total_blocks_n = triton.cdiv(n, BLOCK_N)
        total_programs  = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)
        fp8_fast_accum = False

        c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
        k = _base_quantized_matmul[grid](
            a, b, c, scales, zeros,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            quant_groupsize,
            m, n, k,
            BLOCK_M, BLOCK_N, block_size_k, group_size_m, fp8_fast_accum = fp8_fast_accum,
            num_warps = num_warps, num_stages = num_stages,
        )

        print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c


base_qlinear = base_qlinear.apply





from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


# This is a matmul kernel based on triton.ops.matmul
# It is modified to support rowwise quantized input and global quantized weight
# It's purpose is fused matmul then dequantize
# It does support bias.

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                        num_stages=num_stages, num_warps=num_warps))
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                                                        num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit()
def _custom_quantized_matmul(A, B, C, scales_ptr, zeros_ptr,
                             stride_am, stride_ak,
                             stride_bk, stride_bn,
                             stride_cm, stride_cn,
                             stride_scales_g, stride_scales_n,
                             stride_zeros_g, stride_zeros_n,
                             groupsize,
                             M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             group_size_m: tl.constexpr,
                             fp8_fast_accum: tl.constexpr,
                             SPLIT_K: tl.constexpr,
                             EVEN_K: tl.constexpr):

    a_ptr = A
    b_ptr = B
    c_ptr = C

    pid = tl.program_id(0)
    pid_z = tl.program_id(0)

    total_blocks_m = tl.cdiv(M, BLOCK_M)
    total_blocks_n = tl.cdiv(N, BLOCK_N)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = pid_z*tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):

        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.)
            g_id = k_remaining // (groupsize // BLOCK_K)

            ptr = scales_ptrs + g_id * stride_scales_g
            scales = tl.load(ptr)
            ptr = zeros_ptrs + g_id * stride_zeros_g
            zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K//8) * stride_bk

    acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < n) & (offs_cn[None, :] < n)

    if SPLIT_K == 1:
        tl.store(C, acc, mask=c_mask)
    else:
        tl.atomic_add(C, acc, mask=c_mask)




class custom_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros):

        M, K = a.shape
        _, N = b.shape

        quant_groupsize = 128
        BLOCK_M = 16
        BLOCK_N = 64
        group_size_m = 8
        fp8_fast_accum = True

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
        k = _custom_quantized_matmul[grid](
            a, b, c, scales, zeros,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            quant_groupsize,
            M, N, K,
            group_size_m=group_size_m, fp8_fast_accum=fp8_fast_accum,
        )

        print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c


custom_qlinear = custom_qlinear.apply
def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # Fill with random integers for int32 type
        res = torch.randint(low=-2147483648, high=2147483647, size=(M, N), dtype=dtype, device="cuda")
    else:
        # Fill with normally distributed random values for other types
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res


# from auto_gptq
from typing import Any, Dict, List, Optional, Tuple
def autogptq_post_init(model, use_act_order: bool, max_input_length: Optional[int] = None):
    """
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    device_to_buffers_size = {}

    model_uses_exllama = False
    for name, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
            model_uses_exllama = True
            device = submodule.qweight.device
            if device not in device_to_buffers_size:
                device_to_buffers_size[device] = {
                    "max_dq_buffer_size": 1,
                    "max_inner_outer_dim": 1
                }

            if not use_act_order:
                submodule._use_act_order = False
            else:
                submodule._use_act_order = True

            # Disable this heuristic for detecting act_order, but it could be used instead of the config.
            """
            if submodule.g_idx is None:
                submodule.act_order = False
            elif submodule.g_idx is not None and ((submodule.g_idx == 0).all() or torch.equal(submodule.g_idx.cpu(), torch.tensor([i // submodule.group_size for i in range(submodule.g_idx.shape[0])], dtype=torch.int32))):
                submodule.g_idx = None
                submodule.act_order = False
            else:
                submodule.act_order = True
            """

            device_to_buffers_size[device]["max_dq_buffer_size"] = max(device_to_buffers_size[device]["max_dq_buffer_size"], submodule.qweight.numel() * 8)

            if use_act_order:
                device_to_buffers_size[device]["max_inner_outer_dim"] = max(device_to_buffers_size[device]["max_inner_outer_dim"], submodule.infeatures, submodule.outfeatures)

    if model_uses_exllama:
        # To be honest this is quite ugly, not proud of this.
        try:
            from exllama_kernels import prepare_buffers, set_tuning_params
        except ImportError as e:
            raise ImportError(f"Could not import exllama backend dependencies prepare_buffers, set_tuning_params with the following error: {e}")

        device_to_buffers = {}

        if use_act_order:
            if max_input_length is None:
                max_input_len = EXLLAMA_DEFAULT_MAX_INPUT_LENGTH
            else:
                max_input_len = max_input_length
        else:
            if max_input_length is not None:
                logger.info("Using exllama backend without act-order, the parameter max_input_length was set although not needed, it will be ignored.")
            max_input_len = 1

        for device, buffers_size in device_to_buffers_size.items():
            # The temp_state buffer is required to reorder X in the act-order case.
            # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
            device_to_buffers[device] = {
                "temp_state": torch.zeros((max_input_len, buffers_size["max_inner_outer_dim"]), dtype=torch.float16, device=device),
                "temp_dq": torch.zeros((1, buffers_size["max_dq_buffer_size"]), dtype=torch.float16, device=device),
                "max_dq_buffer_size": buffers_size["max_dq_buffer_size"],
                "max_inner_outer_dim": buffers_size["max_inner_outer_dim"],
            }

        # Buffers need to be persistent to avoid any bug.
        model.device_to_buffers = device_to_buffers

        for device, buffers in model.device_to_buffers.items():
            prepare_buffers(device, buffers["temp_state"], buffers["temp_dq"])

        # Using the default from exllama repo here.
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

        # The buffers need to have been initialized first before calling make_q4.
        for name, submodule in model.named_modules():
            if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
                submodule.post_init()

    ## exllamav2
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for _, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device,0))

    if model_uses_exllamav2:
        from ..nn_modules.qlinear.qlinear_exllamav2 import ExLlamaV2DeviceTensors
        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ExLlamaV2DeviceTensors(device.index, scratch_bytes)

        # have persistent buffers, otherwise we will get OOM
        model.device_tensors = device_tensors

        for _, submodule in model.named_modules():
            if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
                device = submodule.qweight.device
                submodule.post_init(temp_dq = model.device_tensors[device])
    torch.cuda.empty_cache()

    return model


@triton.jit()
def _h100_quantized_matmul(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                             stride_am, stride_ak,
                             stride_bk, stride_bn,
                             stride_cm, stride_cn,
                             stride_scales_g, stride_scales_n,
                             stride_zeros_g, stride_zeros_n,
                             groupsize,
                             m, n, k,
                             block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr,
                             group_size_m: tl.constexpr,
                             fp8_fast_accum: tl.constexpr,):

    pid = tl.program_id(0)

    total_blocks_m = tl.cdiv(m, block_size_m)
    total_blocks_n = tl.cdiv(n, block_size_n)
    total_blocks_k = tl.cdiv(k, block_size_k)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_n = pid_n * block_size_n + tl.arange(0, block_size_n)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_size_n), block_size_n)
    offs_k = tl.arange(0, block_size_k)

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(m,k), strides=(stride_am, stride_ak),
                                offsets=(pid_m*block_size_m, 0), block_shape=(block_size_m, block_size_k),
                                order =(1,0))


    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    acc = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):

        a = tl.load(a_block_ptr, boundary_check=(0,1))
        b = tl.load(b_ptrs)
        g_id = k // (groupsize // block_size_k)

        ptr = scales_ptrs + g_id * stride_scales_g

        scales = tl.load(ptr)
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        if fp8_fast_accum:
            acc = tl.dot(a.to(tl.float8e4nv), b.to(tl.float8e4nv), acc)
        else:
            acc += tl.dot(a,b)

        a_block_ptr = tl.advance(a_block_ptr, (0, block_size_k))
        b_ptrs += (block_size_k//8) * stride_bk

    acc.to(tl.float16)
    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < n) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, acc, mask=c_mask)





class h100_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros):

        m, k = a.shape
        _, n = b.shape

        quant_groupsize = 128
        block_size_m = 16
        block_size_n = 32
        block_size_k = 256
        group_size_m = 8
        num_warps = 4
        num_stages = 4
        total_blocks_m = triton.cdiv(m, block_size_m)
        total_blocks_n = triton.cdiv(n, block_size_n)
        total_programs  = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)
        fp8_fast_accum = False

        c = torch.zeros((m, n), device=a.device, dtype=a.dtype)
        k = _h100_quantized_matmul[grid](
            a, b, c, scales, zeros,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            quant_groupsize,
            m, n, k,
            block_size_m, block_size_n, block_size_k, group_size_m, fp8_fast_accum = fp8_fast_accum,
            num_warps = num_warps, num_stages = num_stages,
        )

        print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c


h100_qlinear = h100_qlinear.apply


if __name__ == '__main__':

    m = 1
    k = 4096
    n = 4096
    groupsize = 128
    g = k // groupsize

    a = make_tensor(m, k, dtype=torch.float16)
    b = make_tensor(k//8, n, dtype=torch.int32)
    c = make_tensor(m, n, dtype=torch.float16)
    zeros = make_tensor(g, n//8, torch.int32)
    scales = make_tensor(g, n, torch.float16)


    # lambda: triton_matmul4(groupsize, a, b, scales, zeros)

    # base = no_autotune(groupsize, a, b, scales, zeros)
    # print(f"{base.shape=}, {base[0][0:4]}")

    # ExllamaV2 (TGIS impl)



    # c = custom_qlinear(a, b, scales, zeros)
    # print(f"{c.shape=}, {c[0][0:4]}")

    small_output = small_qlinear(a, b, scales, zeros)
    # print(f"{small_output.shape=}, {small_output[0][0:4]}")
    #d = base_qlinear(a, b, scales, zeros)

    # print(f"{d.shape=}, {d[0][0:4]}")

    #from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
    # from auto_gptq.modeling._utils import autogptq_post_init
    import torch
    import pdb
    #from gptq import h100_qlinear


    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n', 'k'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            256 * i for i in range(2, 25)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['gptq', 'triton'],
        # Label name for the lines
        line_names=['AutoGPTQ - Triton', "Custom Triton Kernel"],

        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-AutoGPTQ-Triton",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
    def benchmark(n, k, provider):

        nbits = 4
        group_size=128
        #disable_exllama=True
        #disable_exllamav2=False
        #use_triton = False

        # k = 4096
        # n = 4096

        # linear_class = QuantLinear
        #dynamically_import_QuantLinear(
        #disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2,
        #use_triton=use_triton, desc_act=False, group_size=group_size, bits=nbits)
        # def __init__(self, bits: int, groupsize: int, infeatures: int, outfeatures: int, bias: bool):

        linear = QuantLinear(
        #bits=
        nbits,
        #group_size=
        group_size,
        #infeatures=
        k,
        #outfeatures=
        n,
        bias=0,
        )

        device = torch.device('cuda')

        linear.qweight = torch.randint(-100, 100, size=linear.qweight.shape, dtype=torch.int32)
        linear.scales = linear.scales + 0.002

        linear = linear.eval().to(device)
        linear = autogptq_post_init(linear, use_act_order=False)

        m = 1
        groupsize = 128
        g = k // groupsize

        a = make_tensor(m, k, dtype=torch.float16)
        b = make_tensor(k//8, n, dtype=torch.int32)
        # c = make_tensor(m, n, dtype=torch.float16)
        zeros = make_tensor(g, n//8, torch.int32)
        scales = make_tensor(g, n, torch.float16)

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'gptq':
           ms, min_ms, max_ms = triton.testing.do_bench(lambda: no_autotune(groupsize, a, b, scales, zeros), quantiles=quantiles)
        if provider == 'triton':
            #return 0,0,0
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: h100_qlinear(a, b, scales, zeros), quantiles=quantiles)
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path='./')
