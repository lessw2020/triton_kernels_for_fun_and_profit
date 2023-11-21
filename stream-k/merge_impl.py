# code derived from https://github.com/openai/triton/issues/1393


import torch
import triton
import triton.language as tl

from triton.runtime.driver import CudaUtils

torch.manual_seed(2020)

device = torch.cuda.current_device()
cuda_utils = CudaUtils()
total_sm = cuda_utils.get_cuda_utils(device)
print(f"total SMs: {total_sm}")


@triton.jit()
def swizzle_tile(tile_id,
M,N,K,
block_M: tl.constexpr,
block_N: tl.constexpr,
group_M: tl.constexpr):

grid_m = tl.cdiv(M, block_M)
grid_n = tl.ddiv(N,block_N)
# re-order for L2 perf
width = group_M * grid_n
group_id = tile_id // width
group_size = tl.minimum(grid_m - group_id * group_M, group_M)
pid_m = group_id * group_M + (tile_id % group_size)
return pid_m, pid_n
