import torch
import triton
import triton.language as tl


torch.manual_seed(456)

N, d = 16, 8

Q = torch.rand((N, d)) # (16,8)
K = torch.rand((N, d))
V = torch.rand((N, d))

# tile size for matmul, no op bigger than this size can be stored in SRAM
Br = 4
Bc = d

expected_softmax = torch.softmax(Q @ K.T, dim=1)
expected_attention = expected_softmax @ V

# 1st read
S_mat = Q @ K.T
row_max = torch.max(S_mat, dim=1).values[:, None]
# 2nd read
input_safe = S_mat - row_max
softmax_numerator = torch.exp(input_safe)
# 3rd read
softmax_denominator = torch.sum(softmax_numerator, dim=1)[:, None]
# 4th read
naive_softmax = softmax_numerator / softmax_denominator
# final matmul (another read / write)
matmul_result = naive_softmax @ V

assert torch.allclose(naive_softmax, expected_softmax)
assert torch.allclose(matmul_result, expected_attention)

S_mat_for_check = torch.zeros((N, N))

Br = 4
Bc = d
# N = 16
# d = 8
# blocks = 16 x 
# qkv = 16, 8 (N,d)

attn_score = torch.zeros((N, N))

for block_start_Bc in range(0, N, d): # 0, seq_len, head_dim 0 -> 16, step 8
    block_end_Bc = block_start_Bc + d
    Kj = K[block_start_Bc:block_end_Bc,:] # shape Bc x d
    for block_start_Br in range(0, N, Br): 
        block_end_Br = block_start_Br + Br
        Qi = Q[block_start_Br:block_end_Br,:]  # shape Br x d
        # QKt at tile level
        Sij = Qi @ Kj.T # Br * Bc
        attn_score[block_start_Br:block_end_Br, block_start_Bc:block_end_Bc]+= Sij
    


for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc
    Kj = K[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
        Qi = Q[block_start_Br:block_end_Br, :]  # shape Br x d

        # QKt at the tile level
        Sij = Qi @ Kj.T  # shape Br x Bc
        S_mat_for_check[block_start_Br:block_end_Br, block_start_Bc:block_end_Bc] += Sij

print(f"testing attn_score vs ref")
assert torch.allclose(attn_score, Q @ K.T)

'''
O = torch.zeros((N, d))

for block_start_Bc in range(0, N, Bc):  # 0 - 16, step 8
    block_end_Bc = block_start_Bc + Bc
    Kj = K[block_start_Bc:block_end_Bc, :]  # shape Bc x d  8x8... 0,8 then 8,16
    Vj = V[block_start_Bc:block_end_Bc, :]  # shape Bc x d 8x8
    for block_start_Br in range(0, N, Br): # 0-16, step 4
        block_end_Br = block_start_Br + Br
        Qi = Q[block_start_Br:block_end_Br, :]  # shape Br x d 4 x 8  

        # QKt at the tile level
        Sij = Qi @ Kj.T  # shape Br x Bc
        Oi = Sij @ Vj  # shape Br x d
        O[block_start_Br:block_end_Br, :] += Oi
'''


O = torch.zeros((N, d)) # seq * embedding dim

for block_start_Bc in range(0, N, d):
    block_end_Bc = block_start_Bc + d
    Kj = K[block_start_Bc:block_end_Bc, :] # shape Bc x d
    Vj = V[block_start_Bc:block_end_Bc, :]
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
        Qi = Q[block_start_Br:block_end_Br,:]  # shape Br x d

        # QKt at tile level
        Sij = Qi @ Kj.T
        Oi = Sij @ Vj
        O[block_start_Br: block_end_Br, :] += Oi

assert torch.allclose(O, (Q @ K.T) @ V)

# ==========  Flash Attn1.5 =========
O = torch.zeros((N,d))
l = torch.zeros((N,1))
m = torch.full((N,1),-torch.inf)



for block_start_Bc in range(0,N,Bc):
    block_end_Bc = block_start_Bc + Bc
    # load block from input tensor
    Kj = K[block_start_Bc:block_end_Bc,:] # shape Bc x d
    Vj = V[block_start_Bc:block_end_Bc,:] # shape Bc x d
    for block_start_Br in range(0,N, Br):
        block_end_Br = block_start_Br + Br
        # load from global memory
        mi = m[block_start_Br:block_end_Br,:] # shape Br x 1
        li = l[block_start_Br:block_end_Br,:] # shape Br x 1
        Oi = O[block_start_Br:block_end_Br, :] # Br x d
        Qi = Q[block_start_Br:block_end_Br,:] # Br x d

        # QKt at tile level
        Sij = Qi @ Kj.T # Br x Bc

        # max of each row of current block
        mij_hat = torch.max(Sij, dim=1).values[:,None]
        # compute numerator as if we only had data from this block
        pij_hat = torch.exp(Sij - mij_hat)
        lij_hat = torch.sum(pij_hat, dim=1)[:,None]

        # find max of each row for curr block and all prev ones
        mi_new = torch.max(torch.column_stack([mi,mij_hat]), dim=1).values[:,None]
        li_new = torch.exp(mi-mi_new) * li +torch.exp(mij_hat - mi_new) * lij_hat

        Oi = (li * torch.exp(mi-mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @Vj
        #Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

        # line 12, first part before the "+" is the adjustment of the past blocks
        # second part after the "+" is the incorporation of the information from the current block and the matmul for this block
       # Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

        m[block_start_Br:block_end_Br,:] = mi_new # row max
        l[block_start_Br:block_end_Br,:] = li_new # normalizer
        O[block_start_Br:block_end_Br,:] = Oi


assert torch.allclose(O, expected_attention)
print(f"testing first flash attn!")
assert torch.allclose(O, expected_attention)
print(f"Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



# variables outside the for loop represent the global memory
# they are the only ones bigger than what the SRAM can store
O = torch.zeros((N, d))

# For the 2 variables below, they may be removed in a serially executed code (in particular the outter for loop)
# They are needed in parallelized execution where each thread block need to sync its findings with the others
# line 4, l will store the denominator of the softmax for each row
l = torch.zeros((N, 1))
# line 4, m will store the row max (computed progressively, block after block)
m = torch.full((N, 1), -torch.inf)

for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc
    # line 6, load a block from matmul input tensor
    Kj = K[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    Vj = V[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br

        # line 8, load stuff from globabl memory, aka the work of the other thread blocks
        mi = m[block_start_Br:block_end_Br, :]  # shape Br x 1
        li = l[block_start_Br:block_end_Br, :]  # shape Br x 1
        Oi = O[block_start_Br:block_end_Br, :]  # shape Br x d
        Qi = Q[block_start_Br:block_end_Br, :]  # shape Br x d

        # line 9, QKt at the tile level
        Sij = Qi @ Kj.T  # shape Br x Bc

        # line 10, find max of each row of the current loaded block (and only this block)
        mij_hat = torch.max(Sij, dim=1).values[:, None]
        # line 10, compute the softmax numerator like if we only had the data from this block (and nothing before or after)
        pij_hat = torch.exp(Sij - mij_hat)
        # line 10, compute the softmax denominator like if we only had the data from this block (and nothing before or after)
        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        # line 11, find max of each row regarding the current block and all the previous ones we have already visited
        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]
        # line 11, adjusting factor (see online softmax computation above) leveraging the rule of exponentiation
        li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        # line 12, first part before the "+" is the adjustment of the past blocks
        # second part after the "+" is the incorporation of the information from the current block and the matmul for this block
        Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

        # Note that we replace (=) and not update (+=) the global variables like we would do in tilted matmul
        # line 13, save statistics
        m[block_start_Br:block_end_Br, :] = mi_new  # row max
        l[block_start_Br:block_end_Br, :] = li_new  # softmax denominator
        # save attention block to global memory
        O[block_start_Br:block_end_Br, :] = Oi

assert torch.allclose(O, expected_attention)
print(f"testing first flash attn!")
assert torch.allclose(O, expected_attention)
print(f"Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



    