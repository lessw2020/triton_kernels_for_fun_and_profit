# triton_kernels_for_fun_and_profit
Custom kernels in Triton language for accelerating LLMs

First kernel added - a Triton Fused Softmax.    
Currently same speed and numerics as PyTorch Softmax in E2E training, hopefully better tuning will accelerate past PyTorch.

Next up - RMSNorm.  Fwd working, bwd in progress.
