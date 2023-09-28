import torch
import triton
import triton.language as tl

import torch

#M, N, K = 15, 9, 12
M = 15
N = 9
K = 12



A = torch.rand((M, K))
B = torch.rand((K, N))

# guidance from https://github.com/ELS-RD/kernl/blob/main/tutorial/1%20-%20tiled%20matmul.ipynb

# large tile size increase data reuse, but decrease thread-level parallelism;
# small tile size increase thread-level parallelism but reduce data reuse.

# for simplification tile shapes are all multiple of matrix shapes
# otherwise we would need to check matrix bounds and mask out of bounds values by 0s in tiles

def matmul():
    blockm = M //3  # 5
    blockn = N//3  # 3
    blockk = K//2  # 6 
    output = torch.zeros((M,N))

    total_load = 0
    total_writes = 0

    for indexM in range(0,M, blockm):
        startM = indexM
        endM = indexM + blockm

        for indexN in range(0, N, blockn):
            startN = indexN
            endN = indexN + blockn
            accumulator = torch.zeros((blockm, blockn))

            for indexK in range(0,K, blockk):
                startk = indexK
                endk = startk + blockk

                tileA = A[startM:endM,startk:endk]
                total_load += tileA.numel()
                tileB = B[startk:endk, startN:endN]
                total_load += tileB.numel()

                accumulator += tileA @ tileB
            output[startM:endM, startN:endN] = accumulator
            total_writes +=accumulator.numel()
    assert torch.allclose(output, A @ B)
    print("total load from GM:", total_load)
    print("total writes to GM:", total_writes)

matmul()



block_M, block_N, block_K = M // 3, N // 3, K // 2

output = torch.zeros((M, N))
total_load = 0
total_write = 0

for index_M in range(0, M, block_M):
    start_M = index_M
    end_M = index_M + block_M

    for index_N in range(0, N, block_N):
        start_N = index_N
        end_N = index_N + block_N
        accumulator = torch.zeros((block_M, block_N))
        for index_K in range(0, K, block_K):
            start_K = index_K
            end_K = index_K + block_K

            tile_A = A[start_M:end_M, start_K:end_K]
            total_load += tile_A.numel()
            tile_B = B[start_K:end_K, start_N:end_N]
            total_load += tile_B.numel()
            # @ means matmul in numpy and pytorch
            accumulator += tile_A @ tile_B
        output[start_M:end_M, start_N:end_N] = accumulator
        total_write += accumulator.numel()

assert torch.allclose(output, A @ B)
print("total load from GM:", total_load)
print("total write to GM:", total_write)