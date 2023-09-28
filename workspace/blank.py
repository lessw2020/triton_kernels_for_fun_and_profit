import torch
import triton
import triton.language as tl
import time

def naive_softmax(x: torch.Tensor)-> torch.Tensor:
    """ eager mode Softmax"""
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    sm_out = numerator/denominator[:,None]
    return sm_out



def online_softmax(x: torch.Tensor)->torch.Tensor:
    """ online softmax, 2.5x fewer passes than eager algo """
    row_count, col_count = x.shape
    assert x.dim() ==2, " only 2d inputs atm"
    output = torch.zeros_like(x)

    for r in range(row_count):
        row_max = 0 # m 
        normalizer = 0  # l
        for c in range(col_count):
            curr = x[r,c]
            prev_old_max = row_max
            row_max = max(row_max, curr)
            # if row_max > prev_old_max:
            #     print(f"updated row max is now {row_max}, row = {r}")
            normalizer = normalizer * torch.exp(prev_old_max - row_max) + torch.exp(curr - row_max)
        output[r,:] = torch.exp(x[r,:] - row_max) / normalizer
    return output







# ---- simple unit testing ----

sample = torch.tensor([[1,2,3,4,5],[5,4,3,2,1]], dtype=torch.float32, device='cuda')
start = time.perf_counter()
eager_out = naive_softmax(sample)
stop = time.perf_counter()
eager_time = stop-start
start = time.perf_counter()
online_out = online_softmax(sample)
stop = time.perf_counter()
online_time = stop - start
ref_out = torch.softmax(sample, dim=1)

print(f"{eager_out=}\n{online_out=}\n{ref_out=}\n")
print(f"{eager_time=}, {online_time=}")




