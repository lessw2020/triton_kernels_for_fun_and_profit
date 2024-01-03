@triton.jit
def layer_norm_xformers(
    output_ptr,
    a_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    output_row_stride,
    output_col_stride,
    a_row_stride,
    a_col_stride,
    N_SIZE,
    eps,
    HAS_BIAS: tl.constexpr,  # not used, just to make the signature similar to single pass
    IS_RMSNORM: tl.constexpr,  # not used, just to make the signature similar to single pass
    BLOCK_N_SIZE: tl.constexpr,
):
    """
    LayerNorm forward pass for a single feature.
    Requires that a whole row of X is loaded into shared memory -> won't work for large tensors.
    based on:
    https://github.com/facebookresearch/xformers/blob/main/xformers/triton/k_layer_norm.py
    (arg names modified to match other implementation)
    -> only used in benchmarks
    """

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N_SIZE)
    mask = cols < N_SIZE

    x_ptrs = a_ptr + row * a_row_stride + cols * a_col_stride

    x = tl.load(x_ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N_SIZE
    x_zm = tl.where(mask, x - mean, 0.0)
    tl.store(mean_ptr + row, mean)

    x_var = tl.sum(x_zm * x_zm, axis=0) / N_SIZE
    rstd = 1.0 / tl.sqrt(x_var + eps)

    # Normalize
    y = x_zm * rstd
    tl.store(rstd_ptr + row, rstd)

    y = y * w + b
    y_ptrs = output_ptr + row * output_row_stride + cols * output_col_stride
    tl.store(y_ptrs, y, mask=mask)
