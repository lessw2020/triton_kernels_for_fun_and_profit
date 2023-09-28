import torch

torch.manual_seed(2020)

N = 128
d = 64

Q = torch.rand((N,d))
K = torch.rand((N,d))
V = torch.rand((N,d))

ref_softmax = torch.softmax(Q@K.T, dim=1) # N,N
ref_attn = ref_softmax @ V # N,N @ N, d = N,d

Br = 4
Bc = d # all columns since head dims are small (~ 128 or smaller)

assert N % Br ==0, "row blocks must evenly divide into row size"

# HBM stored
Output = torch.zeros((N,d))

for block_start_Br in range(0, N, Br):
    block_end_Br = block_start_Br + Br
    # load a block from HBM
    Qi = Q[block_start_Br:block_end_Br,:]
    # init meta stats and intermediate O storage
    Oi = torch.zeros((Br,d))  # block of rows,  all columns
    li = torch.zeros((Br,1)) # Br x 1...normalizer per row
    mi = torch.full((Br,1), -torch.inf) # Br x1...max per row, init to -inf

    for block_start_Bc in range(0, N, Bc):
        block_end_Bc = block_start_Bc + Bc

         # load K and V input blocks
        Kj=K[block_start_Bc:block_end_Bc,:]
        Vj=V[block_start_Bc:block_end_Bc,:]

        # tile level scores
        Sij=Qi@Kj.T

        # new global max - max of each row of current block (Sij) and previous ones (mi)
        mi_new = torch.max(torch.column_stack([mi, torch.max(Sij, dim=1).values[:,None]]), dim=1).values[:,None]

        # compute local sm numerator
        Pij_local = torch.exp(Sij-mi_new)

        # adjust normalizer based on new max and latest Pij
        li = li * torch.exp(mi-mi_new) + torch.sum(Pij_local, dim=1)[:,None]

        # update output (left side updates prev Oi, right side adds new outputs)
        Oi = Oi * torch.exp(mi-mi_new) + Pij_local@Vj

        mi = mi_new

    Oi = Oi / li  # finalize Oi with normalizer

    # save to HBM
    Output[block_start_Br:block_end_Br, :] = Oi

# compare with reference
print(f"First FA2!")
torch.testing.assert_close(Output, ref_attn)
print(f"Sucess with FA2  !!!!!!!!!!!!!!!!!!!!!!!!")
    

