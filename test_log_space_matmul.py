#!/usr/bin/env python3
"""
Test implementation for log-space matrix multiplication.

This implements signed log-space matrix-vector multiplication:
    result = R @ h
where h is stored as (log|h|, sign(h)) pairs.

The algorithm:
1. Decompose R into log_R_pos = log(max(R, 0)) and log_R_neg = log(max(-R, 0))
2. For each output element, compute contributions in log space
3. Accumulate positive and negative contributions using logsumexp
4. Combine using signed log addition

Author: Erik Garrison
"""

import torch
import torch.nn.functional as F
import math

# Constants
LOG_ZERO = -1e10  # Represents log(0) = -inf, but finite for computation


def to_log_space(x: torch.Tensor, min_log: float = -1e10) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert linear values to (log|x|, sign(x)) representation."""
    sign_x = torch.sign(x)
    sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)  # Treat 0 as positive
    abs_x = torch.abs(x)
    # Use minimum log value for very small numbers to avoid -inf
    log_x = torch.where(abs_x > 0, torch.log(abs_x), torch.full_like(x, min_log))
    return log_x, sign_x


def from_log_space(log_x: torch.Tensor, sign_x: torch.Tensor) -> torch.Tensor:
    """Convert (log|x|, sign(x)) back to linear values."""
    return sign_x * torch.exp(log_x)


def signed_log_add(log_a: torch.Tensor, sign_a: torch.Tensor,
                   log_b: torch.Tensor, sign_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (log|a+b|, sign(a+b)) where a = sign_a * exp(log_a), b = sign_b * exp(log_b).

    This is the key primitive for log-space arithmetic with signed values.
    """
    # Find max and min logs
    max_log = torch.maximum(log_a, log_b)
    min_log = torch.minimum(log_a, log_b)
    diff = min_log - max_log  # Always <= 0

    # Determine which has max log
    a_is_max = log_a >= log_b
    sign_max = torch.where(a_is_max, sign_a, sign_b)
    sign_min = torch.where(a_is_max, sign_b, sign_a)

    same_sign = (sign_max * sign_min) > 0

    # log(exp(max) + exp(min)) = max + log(1 + exp(diff))
    log_same = max_log + torch.log1p(torch.exp(diff))

    # log(exp(max) - exp(min)) = max + log(1 - exp(diff))
    exp_diff = torch.exp(diff)
    # Handle complete cancellation
    cancellation = exp_diff >= 1.0
    log_diff = torch.where(
        cancellation,
        torch.full_like(max_log, LOG_ZERO),
        max_log + torch.log1p(-torch.clamp(exp_diff, max=0.9999999))
    )

    log_result = torch.where(same_sign, log_same, log_diff)
    sign_result = torch.where(cancellation & ~same_sign, torch.ones_like(sign_max), sign_max)

    return log_result, sign_result


def logsumexp_masked(log_vals: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute logsumexp over elements where mask is True.
    Returns LOG_ZERO if no elements are selected.
    """
    # Set masked-out values to -inf
    masked_vals = torch.where(mask, log_vals, torch.full_like(log_vals, float('-inf')))
    # Use torch.logsumexp which handles -inf properly
    result = torch.logsumexp(masked_vals, dim=dim)
    # Replace -inf with LOG_ZERO for downstream computation
    result = torch.where(torch.isinf(result) & (result < 0), torch.full_like(result, LOG_ZERO), result)
    return result


def log_space_matmul_reference(R: torch.Tensor, log_h: torch.Tensor, sign_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of log-space matrix-vector multiplication.

    Args:
        R: Weight matrix [dim_out, dim_in] in LINEAR space
        log_h: Log magnitudes [batch, dim_in]
        sign_h: Signs [batch, dim_in] in {-1, +1}

    Returns:
        (log_result, sign_result): tuple of [batch, dim_out] tensors
    """
    batch_size = log_h.shape[0]
    dim_in = R.shape[1]
    dim_out = R.shape[0]

    # Decompose R into positive and negative parts
    R_pos = torch.clamp(R, min=0)
    R_neg = torch.clamp(-R, min=0)

    # Compute log of R parts (LOG_ZERO where zero)
    log_R_pos = torch.where(R_pos > 0, torch.log(R_pos), torch.full_like(R, LOG_ZERO))
    log_R_neg = torch.where(R_neg > 0, torch.log(R_neg), torch.full_like(R, LOG_ZERO))

    # Output tensors
    log_result = torch.zeros(batch_size, dim_out, device=R.device, dtype=R.dtype)
    sign_result = torch.zeros(batch_size, dim_out, device=R.device, dtype=R.dtype)

    for b in range(batch_size):
        for i in range(dim_out):
            # Accumulate positive and negative contributions
            log_pos_contrib = []
            log_neg_contrib = []

            for j in range(dim_in):
                if R[i, j] > 0:
                    # Positive R: sign of contribution = sign_h[j]
                    log_contrib = log_R_pos[i, j] + log_h[b, j]
                    if sign_h[b, j] > 0:
                        log_pos_contrib.append(log_contrib)
                    else:
                        log_neg_contrib.append(log_contrib)
                elif R[i, j] < 0:
                    # Negative R: sign of contribution = -sign_h[j]
                    log_contrib = log_R_neg[i, j] + log_h[b, j]
                    if sign_h[b, j] < 0:  # -1 * -1 = +1
                        log_pos_contrib.append(log_contrib)
                    else:  # -1 * +1 = -1
                        log_neg_contrib.append(log_contrib)
                # If R[i,j] == 0, no contribution

            # Compute logsumexp for each group
            if log_pos_contrib:
                log_pos = torch.logsumexp(torch.stack(log_pos_contrib), dim=0)
            else:
                log_pos = torch.tensor(LOG_ZERO, device=R.device, dtype=R.dtype)

            if log_neg_contrib:
                log_neg = torch.logsumexp(torch.stack(log_neg_contrib), dim=0)
            else:
                log_neg = torch.tensor(LOG_ZERO, device=R.device, dtype=R.dtype)

            # Combine: result = pos_sum - neg_sum
            # = +1 * exp(log_pos) + (-1) * exp(log_neg)
            log_out, sign_out = signed_log_add(
                log_pos, torch.tensor(1.0, device=R.device),
                log_neg, torch.tensor(-1.0, device=R.device)
            )

            log_result[b, i] = log_out
            sign_result[b, i] = sign_out

    return log_result, sign_result


def log_space_matmul_vectorized(R: torch.Tensor, log_h: torch.Tensor, sign_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized implementation of log-space matrix-vector multiplication.
    This is the version we'll port to CUDA.

    Args:
        R: Weight matrix [dim_out, dim_in] in LINEAR space
        log_h: Log magnitudes [batch, dim_in]
        sign_h: Signs [batch, dim_in] in {-1, +1}

    Returns:
        (log_result, sign_result): tuple of [batch, dim_out] tensors
    """
    batch_size = log_h.shape[0]
    dim_in = R.shape[1]
    dim_out = R.shape[0]

    # Decompose R into positive and negative parts
    R_pos = torch.clamp(R, min=0)
    R_neg = torch.clamp(-R, min=0)

    # Compute log of R parts
    log_R_pos = torch.where(R_pos > 0, torch.log(R_pos), torch.full_like(R, float('-inf')))
    log_R_neg = torch.where(R_neg > 0, torch.log(R_neg), torch.full_like(R, float('-inf')))

    # Expand for batch: [dim_out, dim_in] + [batch, 1, dim_in] -> [batch, dim_out, dim_in]
    log_h_exp = log_h.unsqueeze(1)  # [batch, 1, dim_in]
    sign_h_exp = sign_h.unsqueeze(1)  # [batch, 1, dim_in]

    # Contribution magnitudes: log|R_ij| + log|h_j|
    # [dim_out, dim_in] + [batch, 1, dim_in] -> [batch, dim_out, dim_in]
    log_contrib_from_pos_R = log_R_pos.unsqueeze(0) + log_h_exp  # Where R > 0
    log_contrib_from_neg_R = log_R_neg.unsqueeze(0) + log_h_exp  # Where R < 0

    # Determine final signs of contributions
    # Where R > 0: contrib_sign = sign_h
    # Where R < 0: contrib_sign = -sign_h
    R_is_pos = (R > 0).unsqueeze(0)  # [1, dim_out, dim_in]
    R_is_neg = (R < 0).unsqueeze(0)

    # Create unified contribution tensor
    log_contrib = torch.where(R_is_pos, log_contrib_from_pos_R,
                              torch.where(R_is_neg, log_contrib_from_neg_R,
                                          torch.full_like(log_contrib_from_pos_R, float('-inf'))))

    # Contribution sign
    contrib_sign = torch.where(R_is_pos, sign_h_exp,
                               torch.where(R_is_neg, -sign_h_exp,
                                           torch.zeros_like(sign_h_exp)))

    # Masks for positive and negative contributions
    pos_mask = contrib_sign > 0
    neg_mask = contrib_sign < 0

    # Compute logsumexp for positive contributions
    log_pos_contrib = torch.where(pos_mask, log_contrib, torch.full_like(log_contrib, float('-inf')))
    log_pos = torch.logsumexp(log_pos_contrib, dim=-1)  # [batch, dim_out]

    # Compute logsumexp for negative contributions
    log_neg_contrib = torch.where(neg_mask, log_contrib, torch.full_like(log_contrib, float('-inf')))
    log_neg = torch.logsumexp(log_neg_contrib, dim=-1)  # [batch, dim_out]

    # Replace -inf with LOG_ZERO for signed_log_add
    log_pos = torch.where(torch.isinf(log_pos) & (log_pos < 0),
                          torch.full_like(log_pos, LOG_ZERO), log_pos)
    log_neg = torch.where(torch.isinf(log_neg) & (log_neg < 0),
                          torch.full_like(log_neg, LOG_ZERO), log_neg)

    # Combine: result = pos_sum - neg_sum
    sign_pos = torch.ones_like(log_pos)
    sign_neg = -torch.ones_like(log_neg)

    log_result, sign_result = signed_log_add(log_pos, sign_pos, log_neg, sign_neg)

    return log_result, sign_result


def linear_matmul(R: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Standard linear matrix multiplication for comparison."""
    # R: [dim_out, dim_in], h: [batch, dim_in]
    # result: [batch, dim_out]
    return torch.matmul(h, R.T)


def test_log_space_matmul():
    """Test that log-space matmul matches linear matmul."""
    print("=" * 60)
    print("Testing Log-Space Matrix Multiplication")
    print("=" * 60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test parameters
    batch_size = 4
    dim_in = 8
    dim_out = 6

    # Create test data
    R = torch.randn(dim_out, dim_in, device=device, dtype=torch.float32)
    h_linear = torch.randn(batch_size, dim_in, device=device, dtype=torch.float32)

    # Convert h to log space
    log_h, sign_h = to_log_space(h_linear)

    # Compute using linear method (ground truth)
    result_linear = linear_matmul(R, h_linear)

    # Compute using log-space reference
    log_result_ref, sign_result_ref = log_space_matmul_reference(R, log_h, sign_h)
    result_ref = from_log_space(log_result_ref, sign_result_ref)

    # Compute using log-space vectorized
    log_result_vec, sign_result_vec = log_space_matmul_vectorized(R, log_h, sign_h)
    result_vec = from_log_space(log_result_vec, sign_result_vec)

    # Compare
    print("\n--- Small Matrix Test ---")
    print(f"Matrix R: [{dim_out}, {dim_in}]")
    print(f"Vector h: [{batch_size}, {dim_in}]")

    diff_ref = torch.abs(result_linear - result_ref).max().item()
    diff_vec = torch.abs(result_linear - result_vec).max().item()

    print(f"\nMax absolute error (reference): {diff_ref:.2e}")
    print(f"Max absolute error (vectorized): {diff_vec:.2e}")

    rel_error_ref = (torch.abs(result_linear - result_ref) / (torch.abs(result_linear) + 1e-10)).max().item()
    rel_error_vec = (torch.abs(result_linear - result_vec) / (torch.abs(result_linear) + 1e-10)).max().item()

    print(f"Max relative error (reference): {rel_error_ref:.2e}")
    print(f"Max relative error (vectorized): {rel_error_vec:.2e}")

    # Test with larger matrices
    print("\n--- Large Matrix Test ---")
    dim_in = 512
    dim_out = 512
    batch_size = 32

    R = torch.randn(dim_out, dim_in, device=device, dtype=torch.float32)
    h_linear = torch.randn(batch_size, dim_in, device=device, dtype=torch.float32)
    log_h, sign_h = to_log_space(h_linear)

    result_linear = linear_matmul(R, h_linear)
    log_result_vec, sign_result_vec = log_space_matmul_vectorized(R, log_h, sign_h)
    result_vec = from_log_space(log_result_vec, sign_result_vec)

    print(f"Matrix R: [{dim_out}, {dim_in}]")
    print(f"Vector h: [{batch_size}, {dim_in}]")

    diff_vec = torch.abs(result_linear - result_vec).max().item()
    rel_error_vec = (torch.abs(result_linear - result_vec) / (torch.abs(result_linear) + 1e-10)).max().item()

    print(f"Max absolute error: {diff_vec:.2e}")
    # Compute relative error only for non-tiny values
    nonzero_mask = torch.abs(result_linear) > 1e-6
    if nonzero_mask.any():
        rel_error_vec = (torch.abs(result_linear - result_vec)[nonzero_mask] /
                         torch.abs(result_linear)[nonzero_mask]).max().item()
        print(f"Max relative error (non-tiny): {rel_error_vec:.2e}")
    else:
        print(f"All values too small for relative error")

    # Test numerical stability with very small values
    print("\n--- Numerical Stability Test (small values) ---")
    h_small = h_linear * 1e-30  # Very small values
    log_h_small, sign_h_small = to_log_space(h_small)

    result_linear_small = linear_matmul(R, h_small)
    log_result_small, sign_result_small = log_space_matmul_vectorized(R, log_h_small, sign_h_small)
    result_log_small = from_log_space(log_result_small, sign_result_small)

    print(f"Input scale: 1e-30")
    print(f"Linear result range: [{result_linear_small.min():.2e}, {result_linear_small.max():.2e}]")
    print(f"Log-space result range: [{result_log_small.min():.2e}, {result_log_small.max():.2e}]")

    # Check if linear underflows but log-space preserves values
    linear_zeros = (result_linear_small == 0).sum().item()
    log_zeros = (result_log_small == 0).sum().item()
    print(f"Linear zeros (underflow): {linear_zeros}")
    print(f"Log-space zeros: {log_zeros}")

    # Test numerical stability with very large values
    print("\n--- Numerical Stability Test (large values) ---")
    h_large = h_linear * 1e30  # Very large values
    log_h_large, sign_h_large = to_log_space(h_large)

    result_linear_large = linear_matmul(R, h_large)
    log_result_large, sign_result_large = log_space_matmul_vectorized(R, log_h_large, sign_h_large)
    result_log_large = from_log_space(log_result_large, sign_result_large)

    print(f"Input scale: 1e+30")
    print(f"Linear result range: [{result_linear_large.min():.2e}, {result_linear_large.max():.2e}]")
    print(f"Log-space result range: [{result_log_large.min():.2e}, {result_log_large.max():.2e}]")

    # Check for inf/nan
    linear_inf = torch.isinf(result_linear_large).sum().item()
    log_inf = torch.isinf(result_log_large).sum().item()
    print(f"Linear infs (overflow): {linear_inf}")
    print(f"Log-space infs: {log_inf}")

    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)

    # Test passed if absolute error is small (allowing some error from logsumexp accumulation)
    return diff_vec < 5e-4  # Return True if test passed


def test_deep_recurrence():
    """Test numerical stability through many recurrence steps."""
    print("\n" + "=" * 60)
    print("Testing Deep Recurrence Stability")
    print("=" * 60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dim = 64
    batch_size = 4
    num_steps = 1000

    # Create recurrence matrix with eigenvalues < 1 for stability
    R = torch.randn(dim, dim, device=device) * 0.5 / math.sqrt(dim)

    # Initial hidden state
    h_linear = torch.randn(batch_size, dim, device=device)
    log_h, sign_h = to_log_space(h_linear)

    print(f"Dim: {dim}, Batch: {batch_size}, Steps: {num_steps}")
    print(f"Initial h norm: {h_linear.norm():.4f}")

    # Run linear recurrence
    h_curr_linear = h_linear.clone()
    linear_norms = []
    for t in range(num_steps):
        h_curr_linear = torch.tanh(linear_matmul(R, h_curr_linear))
        if t % 100 == 0:
            linear_norms.append(h_curr_linear.norm().item())

    print(f"\nLinear recurrence norms every 100 steps:")
    for i, n in enumerate(linear_norms):
        print(f"  Step {i*100}: {n:.4e}")

    # Run log-space recurrence
    log_h_curr, sign_h_curr = log_h.clone(), sign_h.clone()
    log_norms = []
    for t in range(num_steps):
        # Matmul in log space
        log_out, sign_out = log_space_matmul_vectorized(R, log_h_curr, sign_h_curr)
        # Convert to linear for tanh
        h_out = from_log_space(log_out, sign_out)
        h_out = torch.tanh(h_out)
        # Convert back to log space
        log_h_curr, sign_h_curr = to_log_space(h_out)

        if t % 100 == 0:
            log_norms.append(h_out.norm().item())

    print(f"\nLog-space recurrence norms every 100 steps:")
    for i, n in enumerate(log_norms):
        print(f"  Step {i*100}: {n:.4e}")

    # Compare final states
    h_final_linear = h_curr_linear
    h_final_log = from_log_space(log_h_curr, sign_h_curr)

    diff = torch.abs(h_final_linear - h_final_log).max().item()
    print(f"\nFinal state max difference: {diff:.4e}")

    print("=" * 60)


if __name__ == "__main__":
    passed = test_log_space_matmul()
    test_deep_recurrence()

    if passed:
        print("\n*** All tests passed! Ready for CUDA implementation. ***")
    else:
        print("\n*** Some tests failed. Debug before CUDA implementation. ***")
