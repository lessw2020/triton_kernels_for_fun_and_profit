import triton
import triton.language as tl
import torch
from triton.runtime.driver import CudaUtils

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
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(m,k), strides=(stride_am, stride_ak),
                                    offsets=(pid_m*block_size_m, 0), block_shape=(block_size_m, block_size_k),
                                    order =(1,0))

    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4


    output = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):

        # a = tl.load(a_block_ptr, boundary_check=(0,1))
        a = tl.load(a_ptrs)
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




class small_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros):

        m, k = a.shape
        _, n = b.shape

        quant_groupsize = 128
        block_size_m = 16
        block_size_n = 32 # [N = 4096 // 32] = 128 blocks
        block_size_k = 256 # [total_blocks_k = 4096 // 4096]
        group_size_m = 8
        num_warps = 4
        num_stages = 8
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



def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # Fill with random integers for int32 type
        res = torch.randint(low=-2147483648, high=2147483647, size=(M, N), dtype=dtype, device="cuda")
    else:
        # Fill with normally distributed random values for other types
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res


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



    # base = no_autotune(groupsize, a, b, scales, zeros)
    # print(f"{base.shape=}, {base[0][0:4]}")

    # ExllamaV2 (TGIS impl)



    # c = custom_qlinear(a, b, scales, zeros)
    # print(f"{c.shape=}, {c[0][0:4]}")
    h100_output = h100_qlinear(a, b, scales, zeros)
    small_output = small_qlinear(a, b, scales, zeros)

    # print(f"{small_output.shape=}, {small_output[0][0:4]}")
    #d = base_qlinear(a, b, scales, zeros)

    # print(f"{d.shape=}, {d[0][0:4]}")

    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n', 'k'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            256 * i for i in range(2, 25)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['bpr', 'reg'],
        # Label name for the lines
        line_names=['Block Pointer', "Regular Pointer"],

        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="block_pointer_performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
    def benchmark(n, k, provider):

        m = 1
        groupsize = 128
        g = k // groupsize

        a = make_tensor(m, k, dtype=torch.float16)
        b = make_tensor(k//8, n, dtype=torch.int32)
        # c = make_tensor(m, n, dtype=torch.float16)
        zeros = make_tensor(g, n//8, torch.int32)
        scales = make_tensor(g, n, torch.float16)

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'bpr':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: h100_qlinear(a, b, scales, zeros), quantiles=quantiles)
        if provider == 'reg':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: small_qlinear(a, b, scales, zeros), quantiles=quantiles)
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path='./perf')
