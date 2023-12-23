# basic vector addition in Triton (similar to the official tutorial)

import torch
import triton
import triton.lang as tl
from triton import cdiv, constexpr   

@triton.jit
def vector_add (a_ptr, b_ptr, c_ptr, num_elements, block_size: constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * block_size
  thread_offsets = block_start + tl.arange(0, block_size)
  mask = offsets < num_elements
  a = tl.load(a_ptr + offsets, mask=mask)
  b = tl.load(b_ptr + offsets, mask=mask)
  output = a + b
  tl.store(c_ptr, output, mask=mask)


