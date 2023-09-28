import torch
import triton
import triton.language as tl



torch.manual_seed(456)

row_count  = 4 
col_count = 16

long_input_vec: torch.Tensor = torch.rand((row_count, col_count))

# torch softmax as a reference
expected_softmax = torch.softmax(long_input_vec, dim=1)

# 1st read, torch max output both indexes and values, we only want the values
# we transpose it to get a vertical tensor
row_max = torch.max(long_input_vec, dim=1).values[:, None]
print("input row max\n", row_max)
# 2nd read
input_safe = long_input_vec - row_max
print("Below we reduce values amplitude, that's the safe part of safe softmax")
print("original 1st row input:\n", long_input_vec[0, :], "safe softmax input 1st row:\n", input_safe[0, :])

softmax_numerator = torch.exp(input_safe)
# 3rd read
normalizer_term = torch.sum(softmax_numerator, dim=1)[:, None]
# 4th read
naive_softmax = softmax_numerator / normalizer_term
print(f"checking naive vs ref softmax..")
assert torch.allclose(naive_softmax, expected_softmax)

online_softmax = torch.zeros_like(long_input_vec)


def online_softmax2(x:torch.Tensor)->torch.Tensor:
    output = torch.zeros_like(x)
    num_rows, num_cols = x.shape

    for r in range(num_rows):
        max_so_far = 0.0
        normalizer = 0.0
        for c in range(num_cols):
            curr = x[r,c]
            prev_max = max_so_far
            if curr > max_so_far:
                max_so_far = curr
                print(f"new_max {max_so_far}")
            normalizer = normalizer * torch.exp(prev_max - max_so_far) + torch.exp(curr - max_so_far)
        # row eval complete
        output[r,:] = torch.exp(x[r,:]-max_so_far) / normalizer

    return output







def f_online_softmax(x: torch.Tensor)-> torch.Tensor:
    online_res = torch.zeros_like(x)
    row_count, col_count = x.shape
    for row in range(row_count):
        row_max = 0.0
        norm_term = 0.0
        print(f"processing row {row}")
        for col in range(col_count):
            val = x[row,col]
            old_row_max = row_max
            row_max = max(old_row_max, val)
            # exponential math = exp(old_row_max - max_row)
            norm_term = norm_term * torch.exp(old_row_max - row_max) + torch.exp(val - row_max) 
            if old_row_max != row_max:
                print(f"new max found. curr_max = {row_max}, denom = {norm_term}")
            
        online_res[row,:] = torch.exp(x[row,:]-row_max)/norm_term
    return online_res

online_out = f_online_softmax(long_input_vec)
online_out2 = online_softmax2(long_input_vec)

print(f"checking online vs expected")
torch.testing.assert_close(online_out2,expected_softmax)

for row in range(row_count):
    row_max = 0.0
    normalizer_term = 0.0
    print("--- new row ---")
    for col in range(col_count):  # scalar level iteration
        val = long_input_vec[row, col]
        old_row_max = row_max
        row_max = max(old_row_max, val)
        # np.exp(old_max_row - max_row) is the adjustment factor of our precedent normalizer term,
        # after this multiplication it's like we had always substracted row_max up to this point
        # instead of old_row_max
        normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(val - row_max)
        if old_row_max != row_max:
            print("new max discovered")
        print(f"current row max: {row_max}, denominator: {normalizer_term}")

    # leverage our 2 statistics
    online_softmax[row, :] = torch.exp(long_input_vec[row, :] - row_max) / normalizer_term

assert torch.allclose(online_softmax, expected_softmax)
