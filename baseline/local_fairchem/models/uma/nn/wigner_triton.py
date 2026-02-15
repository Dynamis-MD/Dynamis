"""
Wigner BMM with Autograd Support - Standalone Triton Kernels

This module provides:
1. Fast forward kernel (extracted from torch.compile)
2. Fast backward kernels (grad_wigner, grad_x)
3. Full autograd integration via torch.autograd.Function
4. torch.compile compatibility via torch.library.triton_op

Usage:
    from wigner_triton import wigner_bmm
    output = wigner_bmm(wigner, x)  # Supports autograd
    output.backward(grad_output)

torch.compile Integration:
- Uses torch.library.triton_op to register as a proper PyTorch op
- torch.compile can see into the implementation for optimization
- Avoids graph breaks and enables fusion with surrounding ops
"""

import torch
import triton
import triton.language as tl
from torch.autograd import Function

# Import torch.library APIs for torch.compile compatibility
try:
    from torch.library import triton_op, wrap_triton
    TRITON_OP_AVAILABLE = True
except ImportError:
    TRITON_OP_AVAILABLE = False
    # Fallback: define no-op wrappers
    def wrap_triton(kernel):
        return kernel


# ==============================================================================
# FORWARD KERNEL: wigner @ x -> output
# Shape: [B, 9, 9] @ [B, 9, 128] -> [B, 9, 128]
# ==============================================================================

@triton.jit
def _wigner_bmm_fwd_kernel(
    A,  # wigner: [B, 9, 9]
    B,  # x: [B, 9, N]
    out_ptr,  # output: [B, 9, N]
    stride_aq, stride_am, stride_ak,
    stride_bq, stride_bk, stride_bn,
    stride_oq, stride_om, stride_on, # Output strides
    B_dim,  # batch size
    # Compile-time constants
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 16,
    GROUP_M: tl.constexpr = 8,
    M: tl.constexpr = 9,
    N: tl.constexpr = 128, # Default, but can be overridden
    K: tl.constexpr = 9,
):
    """Forward: output = wigner @ x"""
    pid = tl.program_id(0)
    
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_pid_mn = grid_m * grid_n
    
    idx_q = pid // num_pid_mn
    pid_mn = pid % num_pid_mn
    
    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    ram = rm % M
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    
    A_ptr = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
    B_ptr = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A_ptr, mask=rk[None, :] < k, other=0.)
        b = tl.load(B_ptr, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=False)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    
    # Use output strides
    out_ptr_base = out_ptr + idx_q * stride_oq
    # indices: [m, n]
    # offset = m * stride_om + n * stride_on
    out_offset = idx_m * stride_om + idx_n * stride_on
    tl.store(out_ptr_base + out_offset, acc, mask)


# ==============================================================================
# BACKWARD KERNEL 1: grad_x = wigner.T @ grad_output
# Shape: [B, 9, 9].T @ [B, 9, 128] -> [B, 9, 128]
# ==============================================================================

@triton.jit
def _wigner_bmm_bwd_x_kernel(
    wigner_T,  # wigner transposed: [B, 9, 9] but accessed as T
    grad_out,  # grad_output: [B, 9, 128]
    grad_x_ptr,  # output: [B, 9, 128]
    stride_wq, stride_wm, stride_wk,  # wigner strides (will access transposed)
    stride_gq, stride_gm, stride_gn,  # grad_output strides
    B_dim,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 16,
    GROUP_M: tl.constexpr = 8,
    M: tl.constexpr = 9,
    N: tl.constexpr = 128,
    K: tl.constexpr = 9,
):
    """grad_x = wigner.T @ grad_output"""
    pid = tl.program_id(0)
    
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_pid_mn = grid_m * grid_n
    
    idx_q = pid // num_pid_mn
    pid_mn = pid % num_pid_mn
    
    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    # wigner.T access: swap m and k indices
    # wigner_T[b, m, k] = wigner[b, k, m]
    A_ptr = wigner_T + (rk[None, :] * stride_wm + rm[:, None] * stride_wk + idx_q * stride_wq)
    G_ptr = grad_out + (rk[:, None] * stride_gm + rn[None, :] * stride_gn + idx_q * stride_gq)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A_ptr, mask=rk[None, :] < k, other=0.)
        g = tl.load(G_ptr, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, g, allow_tf32=False)
        A_ptr += BLOCK_K * stride_wm
        G_ptr += BLOCK_K * stride_gm
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    
    # Use grad_out strides for grad_x (same shape N)
    # Output is grad_x [B, 9, N]
    # stride_gq, stride_gm, stride_gn are for grad_output [B, 9, N]
    # grad_x has same shape, assume contiguous or same stride? 
    # Better to assume contiguous grad_x output or pass specific stride.
    # For now, let's assume contiguous output and compute stride from N.
    # stride_xn = 1, stride_xm = N, stride_xq = 9*N
    
    out_stride_n = 1
    out_stride_m = N
    out_stride_q = M * N
    
    xindex = idx_n * out_stride_n + idx_m * out_stride_m + idx_q * out_stride_q
    tl.store(grad_x_ptr + tl.broadcast_to(xindex, acc.shape), acc, mask)


# ==============================================================================
# BACKWARD KERNEL 2: grad_wigner = grad_output @ x.T
# Shape: [B, 9, 128] @ [B, 128, 9] -> [B, 9, 9]
# ==============================================================================

@triton.jit
def _wigner_bmm_bwd_w_kernel(
    grad_out,  # grad_output: [B, 9, 128]
    x_T,  # x transposed: [B, 9, 128] accessed as [B, 128, 9]
    grad_w_ptr,  # output: [B, 9, 9]
    stride_gq, stride_gm, stride_gk,  # grad_output strides
    stride_xq, stride_xk, stride_xn,  # x strides (K=128, N=9)
    B_dim,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 128,
    GROUP_M: tl.constexpr = 8,
    M: tl.constexpr = 9,
    N: tl.constexpr = 9,
    K: tl.constexpr = 128,
):
    """grad_wigner = grad_output @ x.T"""
    pid = tl.program_id(0)
    
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_pid_mn = grid_m * grid_n
    
    idx_q = pid // num_pid_mn
    pid_mn = pid % num_pid_mn
    
    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    # grad_out: [B, 9, 128], x.T: [B, 128, 9] (via transposed access)
    G_ptr = grad_out + (rm[:, None] * stride_gm + rk[None, :] * stride_gk + idx_q * stride_gq)
    X_ptr = x_T + (rk[:, None] * stride_xk + rn[None, :] * stride_xn + idx_q * stride_xq)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Single iteration since K=128 == BLOCK_K
    g = tl.load(G_ptr, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.)
    x = tl.load(X_ptr, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.)
    acc += tl.dot(g, x, allow_tf32=False)
    
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    
    out_stride = 9 * 9
    xindex = idx_n + 9 * idx_m + out_stride * idx_q
    tl.store(grad_w_ptr + tl.broadcast_to(xindex, acc.shape), acc, mask)


# ==============================================================================
# AUTOGRAD FUNCTION (Legacy - still used as fallback)
# ==============================================================================

class WignerBMMFunction(Function):
    @staticmethod
    def forward(ctx, wigner, x):
        """
        Forward pass: output = wigner @ x
        
        Args:
            wigner: [B, 9, 9] Wigner rotation matrices
            x: [B, 9, 128] input features
            
        Returns:
            output: [B, 9, 128]
        """
        B = wigner.shape[0]
        
        # Ensure contiguous
        wigner = wigner.contiguous()
        x = x.contiguous()
        
        C = x.shape[2]
        
        # Save for backward
        ctx.save_for_backward(wigner, x)
        
        # Allocate output
        output = torch.empty(B, 9, C, device=x.device, dtype=x.dtype)
        
        # Launch forward kernel
        # Grid: (batch * blocks, 1, 1).
        # We need to handle C being potentially large or small.
        # BLOCK_N=128. If C=256, grid_n=2. 
        grid_m = (9 + 16 - 1) // 16
        grid_n = (C + 128 - 1) // 128
        num_blocks = grid_m * grid_n
        grid = (B * num_blocks, 1, 1)
        
        _wigner_bmm_fwd_kernel[grid](
            wigner, x, output,
            wigner.stride(0), wigner.stride(1), wigner.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            B,
            N=C, # Pass N=C
            num_warps=4, num_stages=3,
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass (hybrid strategy for optimal performance):
            grad_x = wigner.T @ grad_output     [B,9,9].T @ [B,9,128] -> [B,9,128]  (Triton)
            grad_wigner = grad_output @ x.T     [B,9,128] @ [B,128,9] -> [B,9,9]    (cuBLAS)
        
        Autotune results show:
        - grad_x shape: Triton is faster (0.33ms vs cuBLAS)
        - grad_wigner shape: cuBLAS is faster (0.40ms vs 0.68ms Triton)
        """
        wigner, x = ctx.saved_tensors
        B = wigner.shape[0]
        
        grad_output = grad_output.contiguous()
        
        C = grad_output.shape[2]
        
        # grad_x = wigner.T @ grad_output (use Triton - faster for this shape)
        grad_x = torch.empty(B, 9, C, device=x.device, dtype=x.dtype)
        
        grid_m = (9 + 16 - 1) // 16
        grid_n = (C + 128 - 1) // 128
        num_blocks = grid_m * grid_n
        grid = (B * num_blocks, 1, 1)
        _wigner_bmm_bwd_x_kernel[grid](
            wigner, grad_output, grad_x,
            wigner.stride(0), wigner.stride(1), wigner.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            B,
            N=C, # Pass N=C
            num_warps=4, num_stages=3,
        )
        
        # grad_wigner = grad_output @ x.T (use cuBLAS - faster for this shape)
        grad_wigner = torch.bmm(grad_output, x.transpose(-1, -2))
        
        return grad_wigner, grad_x


# ==============================================================================
# torch.library.triton_op INTEGRATION (for torch.compile compatibility)
# ==============================================================================

if TRITON_OP_AVAILABLE:
    # Forward op - visible to torch.compile
    @triton_op("uma::wigner_bmm_fwd", mutates_args={})
    def _wigner_bmm_forward_op(wigner: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass wrapped for torch.compile visibility."""
        B = wigner.shape[0]
        wigner = wigner.contiguous()
        x = x.contiguous()
        C = x.shape[2]
        
        output = torch.empty(B, 9, C, device=x.device, dtype=x.dtype)
        
        grid_m = (9 + 16 - 1) // 16
        grid_n = (C + 128 - 1) // 128
        num_blocks = grid_m * grid_n
        
        def grid_fn(meta):
            return (B * num_blocks,)
        
        # Wrap triton kernel for torch.compile visibility
        # IMPORTANT: Pass ALL constexpr arguments explicitly for wrap_triton
        wrap_triton(_wigner_bmm_fwd_kernel)[grid_fn](
            wigner, x, output,
            wigner.stride(0), wigner.stride(1), wigner.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            B,
            # All constexpr arguments must be passed
            BLOCK_M=16, BLOCK_N=128, BLOCK_K=16, GROUP_M=8,
            M=9, N=C, K=9,
            num_warps=4, num_stages=3,
        )
        
        return output

    # Backward implementation for grad_x using Triton
    @triton_op("uma::wigner_bmm_bwd_x", mutates_args={})
    def _wigner_bmm_backward_x_op(wigner: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """Compute grad_x = wigner.T @ grad_output using Triton."""
        B = wigner.shape[0]
        grad_output = grad_output.contiguous()
        C = grad_output.shape[2]
        
        grad_x = torch.empty(B, 9, C, device=wigner.device, dtype=wigner.dtype)
        
        grid_m = (9 + 16 - 1) // 16
        grid_n = (C + 128 - 1) // 128
        num_blocks = grid_m * grid_n
        
        def grid_fn(meta):
            return (B * num_blocks,)
        
        # All constexpr arguments must be passed for wrap_triton
        wrap_triton(_wigner_bmm_bwd_x_kernel)[grid_fn](
            wigner, grad_output, grad_x,
            wigner.stride(0), wigner.stride(1), wigner.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            B,
            BLOCK_M=16, BLOCK_N=128, BLOCK_K=16, GROUP_M=8,
            M=9, N=C, K=9,
            num_warps=4, num_stages=3,
        )
        
        return grad_x

    # Backward function for autograd
    def _wigner_bmm_backward(ctx, grad_output):
        """Backward for uma::wigner_bmm_fwd."""
        wigner, x = ctx.saved_tensors
        
        # grad_x = wigner.T @ grad_output (Triton)
        grad_x = _wigner_bmm_backward_x_op(wigner, grad_output)
        
        # grad_wigner = grad_output @ x.T (cuBLAS - faster for this shape)
        grad_wigner = torch.bmm(grad_output.contiguous(), x.transpose(-1, -2))
        
        return grad_wigner, grad_x

    # Setup context to save tensors for backward
    def _wigner_bmm_setup_context(ctx, inputs, output):
        wigner, x = inputs
        ctx.save_for_backward(wigner, x)

    # Register autograd for the forward op
    _wigner_bmm_forward_op.register_autograd(_wigner_bmm_backward, setup_context=_wigner_bmm_setup_context)


def wigner_bmm(wigner: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Fast Wigner BMM with autograd support.
    
    This is a drop-in replacement for torch.bmm(wigner, x).
    
    When torch.library.triton_op is available (PyTorch 2.3+), this uses
    the registered triton_op for torch.compile compatibility. Otherwise,
    falls back to the WignerBMMFunction autograd.
    
    Args:
        wigner: [B, 9, 9] Wigner rotation matrices
        x: [B, 9, 128] input features
        
    Returns:
        output: [B, 9, 128] = wigner @ x
    """
    if TRITON_OP_AVAILABLE:
        return _wigner_bmm_forward_op(wigner, x)
    else:
        return WignerBMMFunction.apply(wigner, x)


# ==============================================================================
# TESTS
# ==============================================================================

def test_forward():
    """Test forward pass."""
    print("Testing forward pass...")
    B = 35000
    
    wigner = torch.randn(B, 9, 9, device="cuda", dtype=torch.float32)
    x = torch.randn(B, 9, 128, device="cuda", dtype=torch.float32)
    
    ref = torch.bmm(wigner, x)
    out = wigner_bmm(wigner, x)
    
    diff = (ref - out).abs().max().item()
    print(f"  Max diff: {diff:.2e}")
    assert diff < 1e-4, f"Forward test failed: diff={diff}"
    print("  ✓ Forward passed!")


def test_backward():
    """Test backward pass."""
    print("Testing backward pass...")
    B = 1000  # Smaller for gradient check
    
    # Use fp32 - Triton tl.dot doesn't support fp64
    wigner = torch.randn(B, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True)
    x = torch.randn(B, 9, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    
    # Reference backward
    ref_out = torch.bmm(wigner, x)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    ref_grad_w = wigner.grad.clone()
    ref_grad_x = x.grad.clone()
    
    # Reset grads
    wigner.grad = None
    x.grad = None
    
    # Our backward
    out = wigner_bmm(wigner, x)
    out.backward(grad_out)
    
    diff_w = (ref_grad_w - wigner.grad).abs().max().item()
    diff_x = (ref_grad_x - x.grad).abs().max().item()
    
    print(f"  grad_wigner max diff: {diff_w:.2e}")
    print(f"  grad_x max diff: {diff_x:.2e}")
    
    assert diff_w < 1e-4, f"grad_wigner test failed: diff={diff_w}"
    assert diff_x < 1e-4, f"grad_x test failed: diff={diff_x}"
    print("  ✓ Backward passed!")


def benchmark():
    """Benchmark forward and backward."""
    import time
    B = 35000
    
    wigner = torch.randn(B, 9, 9, device="cuda", dtype=torch.float32, requires_grad=True)
    x = torch.randn(B, 9, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    
    # Warmup
    for _ in range(10):
        out = wigner_bmm(wigner, x)
        out.sum().backward()
        wigner.grad = None
        x.grad = None
    torch.cuda.synchronize()
    
    # Benchmark forward
    N = 100
    start = time.time()
    for _ in range(N):
        out = wigner_bmm(wigner, x)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start) / N * 1000
    
    # Benchmark forward + backward
    start = time.time()
    for _ in range(N):
        out = wigner_bmm(wigner, x)
        out.sum().backward()
        wigner.grad = None
        x.grad = None
    torch.cuda.synchronize()
    total_time = (time.time() - start) / N * 1000
    
    print(f"\n=== Triton Kernel Performance ===")
    print(f"Forward:  {fwd_time:.3f} ms")
    print(f"Forward+Backward: {total_time:.3f} ms")
    print(f"Backward (estimated): {total_time - fwd_time:.3f} ms")
    
    # Compare with torch.bmm
    wigner2 = wigner.detach().requires_grad_(True)
    x2 = x.detach().requires_grad_(True)
    
    start = time.time()
    for _ in range(N):
        out = torch.bmm(wigner2, x2)
    torch.cuda.synchronize()
    ref_fwd_time = (time.time() - start) / N * 1000
    
    start = time.time()
    for _ in range(N):
        out = torch.bmm(wigner2, x2)
        out.sum().backward()
        wigner2.grad = None
        x2.grad = None
    torch.cuda.synchronize()
    ref_total_time = (time.time() - start) / N * 1000
    
    print(f"\n=== torch.bmm Performance ===")
    print(f"Forward:  {ref_fwd_time:.3f} ms")
    print(f"Forward+Backward: {ref_total_time:.3f} ms")
    
    print(f"\n=== Speedup ===")
    print(f"Forward: {ref_fwd_time/fwd_time:.2f}x")
    print(f"Total:   {ref_total_time/total_time:.2f}x")


if __name__ == "__main__":
    test_forward()
    test_backward()
    benchmark()
