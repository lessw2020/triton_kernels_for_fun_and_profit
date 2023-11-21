# holding
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    # pid_m = pid // tl.cdiv(M, BLOCK_SIZE_M)
    # pid_n = pid % tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
    # see above `Pointer Arithmetics` section for details
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    for k in range(0, K, BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # tl.store(scratchpad + tl.reshape(tl.arange(0, BLOCK_SIZE_M * BLOCK_SIZE_K), (BLOCK_SIZE_M, BLOCK_SIZE_K)), a)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # you can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b, activation=""):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c



@triton.jit
def streamk_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    scratchpad,
    locks,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
):
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_iters = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N) * iters_per_tile
    iters_per_cta = tl.cdiv(total_iters, tl.num_programs(0))
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    itr = pid * iters_per_cta
    iter_end = itr + iters_per_cta

    GROUP_SIZE_M = 8
    while itr < iter_end and itr < total_iters:
        tile_idx = itr // iters_per_tile
        tile_iter = tile_idx * iters_per_tile
        tile_iter_end = tile_iter + iters_per_tile

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_idx // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        # pid_m = first_pid_m + (tile_idx % group_size_m)
        # pid_n = (tile_idx % num_pid_in_group) // group_size_m
        pid_m = tile_idx % num_pid_m
        pid_n = tile_idx // num_pid_m
        pid_k = itr - tile_iter

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        local_iter = pid_k
        local_iter_end = min(iter_end, tile_iter_end) - tile_iter

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, local_iter_end - local_iter):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        acc = acc.to(tl.float16)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        tile_started = (itr == tile_iter)
        tile_ended = (iter_end >= tile_iter_end)

        scratch_off = pid * BLOCK_SIZE_M * BLOCK_SIZE_N
        offs_scratch= tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]

        if USE_ATOMICS:
            tl.atomic_add(c_ptr + c_offs, acc, c_mask)
        else:
            if not tile_started:
                tl.store(scratchpad + scratch_off + offs_scratch, acc, c_mask)
                tl.atomic_xchg(locks + pid, 1)
            else:
                if not tile_ended:
                    cta_end = tl.cdiv(tile_iter_end, iters_per_cta)
                    cas = pid + 1
                    while cas < cta_end:
                        while tl.atomic_cas(locks + cas, 1, 2) != 1:
                            pass
                        acc += tl.load(scratchpad + cas * BLOCK_SIZE_M * BLOCK_SIZE_N + offs_scratch, c_mask)
                        cas += 1

                tl.store(c_ptr + c_offs, acc, c_mask)

        itr = tile_iter_end


torch.set_printoptions(sci_mode=False)


def streamk_matmul(a, b):
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64

    USE_ATOMICS = False

    # allocates output
    if not USE_ATOMICS:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    # 1D launch kernel where each block gets its own program.
    # grid = lambda META: (
    #     triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    # )
    grid = lambda META: (
       108,
    )
    scratchpad = torch.empty(108, BLOCK_SIZE_M, BLOCK_SIZE_N, device=a.device, dtype=a.dtype)
    locks = torch.zeros(108, device='cuda', dtype=torch.int32)

    streamk_kernel[grid](
        a, b, c,
        M, N, K,
        scratchpad,
        locks,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_ATOMICS=USE_ATOMICS,
        num_stages=3, num_warps=8,
    )
    return c


torch.set_printoptions(sci_mode=False)


from triton.testing import do_bench
torch.manual_seed(0)

def get_flops(M, N, K, fn=streamk_matmul):
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    check_correct(A, B, fn)
    # fn(A, B)
    # ms = 1
    ms = do_bench(lambda: fn(A, B))# [0]
    print(f"{A.shape[0]}: {A.shape[0] * A.shape[1] * B.shape[1] * 2 / ms / 1e9}")

def check_correct(A, B, fn=streamk_matmul):
    out_triton = fn(A, B)
    out = torch.mm(A, B)
    torch.testing.assert_close(out, out_triton, atol=3e-1, rtol=1e-1)

import triton.ops as ops
for N in range(2048, 4096, 256):
    get_flops(N, N, N, ops.matmul)
    get_flops(N, N, N, streamk_matmul)
    print()
