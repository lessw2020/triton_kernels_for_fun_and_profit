import torch
import triton.language as tl
import triton




def cdiv(x,y):
    return (x+y-1)//y
# 5 10

res = cdiv(11,5)
print(res)

tres = cdiv(22,5)
print(tres)


M = 1024
N = 768
K = 128

print(f"C [{M}x{N}] = A [{M}x{K}] * B [{K}x{N}]")

a = torch.rand(M, K)
b = torch.rand(K, N)
c = torch.rand(M, N)

# programs work at the tile level, below are their dimensions for each axis
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32

# group is a special concept to speed up matmul in this tutorial
GROUP_SIZE_M = 2

# # of programs to run on each C axis (each program will iterate over K)
num_pid_m = cdiv(M, BLOCK_SIZE_M)  # number of `program`s in M dimension, rounded to the nearest bigger integer
num_pid_n = cdiv(N, BLOCK_SIZE_N)  # number of `program`s in N dimension, rounded to the nearest bigger integer

print("num_pid_m:", num_pid_m)
print("num_pid_n:", num_pid_n)
print("num_pid_n * GROUP_SIZE_M =", num_pid_n * GROUP_SIZE_M)

num_pid_m = cdiv(M, BLOCK_SIZE_M)
num_pid_n = cdiv(N, BLOCK_SIZE_N)
print(f"{num_pid_m=}, {num_pid_n=}")

nb_programs = num_pid_m * num_pid_n  # number of programs to launch
print("nb programs to launch:", nb_programs)

num_pid_in_group = GROUP_SIZE_M * num_pid_n
assert num_pid_n * GROUP_SIZE_M <= M
print(num_pid_in_group)