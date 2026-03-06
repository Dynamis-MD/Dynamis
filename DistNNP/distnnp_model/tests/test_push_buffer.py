import pytest
import torch
from distnnp_model.distnnp_model_impl import push_buffer_create

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Basic Gather Test
# Manual Seeds Make this Deterministic
# Feature dim 10 to force the scalar kernel
@pytest.mark.parametrize("seed", range(50))
def test_basic_gather(seed):
    torch.manual_seed(seed)
    N, F = 100, 10
    num_indices = 50

    inp = torch.randn(N, F, dtype=torch.float32, device="cuda")
    buf = torch.zeros(num_indices, F, dtype=torch.float32, device="cuda")
    locs = torch.randint(0, N, (num_indices,), dtype=torch.int64, device="cuda")

    result = push_buffer_create(inp, buf, locs)

    expected = inp[locs]
    assert torch.allclose(buf, expected)
    assert torch.allclose(result, expected)

# Same as basic gather but with guarenteed duplicate indices
# Feature dim 10 to force the scalar kernel
@pytest.mark.parametrize("seed", range(50))
def test_duplicate_indices(seed):
    torch.manual_seed(seed)
    N, F = 100, 10
    num_indices = 50

    inp = torch.randn(N, F, dtype=torch.float32, device="cuda")
    buf = torch.zeros(num_indices, F, dtype=torch.float32, device="cuda")

    # Generate at half size and repeat for repeated indices
    locs = torch.randint(0, N, (num_indices // 2,), dtype=torch.int64, device="cuda").repeat(2)

    result = push_buffer_create(inp, buf, locs)

    expected = inp[locs]
    assert torch.allclose(buf, expected)
    assert torch.allclose(result, expected)

# Same as basic gather but only for a single row
# Feature dim 10 to force the scalar kernel
@pytest.mark.parametrize("seed", range(50))
def test_single_row(seed):
    torch.manual_seed(seed)
    N, F = 100, 10
    num_indices = 1

    inp = torch.randn(N, F, dtype=torch.float32, device="cuda")
    buf = torch.zeros(num_indices, F, dtype=torch.float32, device="cuda")
    locs = torch.randint(0, N, (num_indices,), dtype=torch.int64, device="cuda")

    result = push_buffer_create(inp, buf, locs)

    expected = inp[locs]
    assert torch.allclose(buf, expected)
    assert torch.allclose(result, expected)

# Basic gather that gathers all rows in order
# This is essentially a copy
# Feature dim 10 to force the scalar kernel
@pytest.mark.parametrize("seed", range(50))
def test_all_rows_in_order(seed):
    torch.manual_seed(seed)
    N, F = 100, 10
    num_indices = N

    inp = torch.randn(N, F, dtype=torch.float32, device="cuda")
    buf = torch.zeros(num_indices, F, dtype=torch.float32, device="cuda")
    locs = torch.arange(0, N, dtype=torch.int64, device="cuda")

    result = push_buffer_create(inp, buf, locs)

    assert torch.allclose(buf, inp)
    assert torch.allclose(result, inp)

# Basic gather that gathers all rows in reverse
# This is essentially a reverse copy
# Feature dim 10 to force the scalar kernel
@pytest.mark.parametrize("seed", range(50))
def test_all_rows_reversed(seed):
    torch.manual_seed(seed)
    N, F = 100, 10
    num_indices = N

    inp = torch.randn(N, F, dtype=torch.float32, device="cuda")
    buf = torch.zeros(num_indices, F, dtype=torch.float32, device="cuda")
    locs = torch.arange(N - 1, -1, -1, dtype=torch.int64, device="cuda")

    result = push_buffer_create(inp, buf, locs)

    expected = inp.flip(0)
    assert torch.allclose(buf, expected)
    assert torch.allclose(result, expected)

# Basic Gather Test that forces Vec4
@pytest.mark.parametrize("feature_dim", [4, 16, 64, 128, 256])
@pytest.mark.parametrize("seed", range(50))
def test_gather_vec4(seed, feature_dim):
    torch.manual_seed(seed)
    N = 100
    num_indices = 50

    inp = torch.randn(N, feature_dim, dtype=torch.float32, device="cuda")
    buf = torch.zeros(num_indices, feature_dim, dtype=torch.float32, device="cuda")
    locs = torch.randint(0, N, (num_indices,), dtype=torch.int64, device="cuda")

    result = push_buffer_create(inp, buf, locs)

    expected = inp[locs]
    assert torch.allclose(buf, expected)
    assert torch.allclose(result, expected)

# Test the early return path when locs is empty
def test_empty_indices():
    N, F = 100, 10

    inp = torch.randn(N, F, dtype=torch.float32, device="cuda")
    locs = torch.empty(0, dtype=torch.int64, device="cuda")
    buf = torch.zeros(0, F, dtype=torch.float32, device="cuda")

    result = push_buffer_create(inp, buf, locs)

    assert result.shape == (0, F)
    assert result.data_ptr() == buf.data_ptr()
    
# Like Basic Gather but for very large input
# Feature dim 65 to force the scalar kernel
# Less seeds on this one because it is already large
@pytest.mark.parametrize("seed", range(5))
def test_large_input(seed):
    torch.manual_seed(seed)
    N, F = 100000, 65
    num_indices = 50000

    inp = torch.randn(N, F, dtype=torch.float32, device="cuda")
    buf = torch.zeros(num_indices, F, dtype=torch.float32, device="cuda")
    locs = torch.randint(0, N, (num_indices,), dtype=torch.int64, device="cuda")

    result = push_buffer_create(inp, buf, locs)

    expected = inp[locs]
    assert torch.allclose(buf, expected)
    assert torch.allclose(result, expected)

# Like Basic Gather but for very large input
# Feature dim 64 to force the vec4 kernel
# Less seeds on this one because it is already large
@pytest.mark.parametrize("seed", range(5))
def test_large_input_vec4(seed):
    torch.manual_seed(seed)
    N, F = 100000, 64
    num_indices = 50000

    inp = torch.randn(N, F, dtype=torch.float32, device="cuda")
    buf = torch.zeros(num_indices, F, dtype=torch.float32, device="cuda")
    locs = torch.randint(0, N, (num_indices,), dtype=torch.int64, device="cuda")

    result = push_buffer_create(inp, buf, locs)

    expected = inp[locs]
    assert torch.allclose(buf, expected)
    assert torch.allclose(result, expected)

# Rejects CPU input tensor
# Should raise RuntimeError from CHECK_INPUT macro
def test_rejects_cpu_input():
    inp = torch.randn(4, 3, dtype=torch.float32, device="cpu")
    buf = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        push_buffer_create(inp, buf, locs)

# Rejects non-contiguous input tensor
# Should raise RuntimeError from CHECK_INPUT macro
def test_rejects_non_contiguous():
    inp = torch.randn(4, 3, dtype=torch.float32, device="cuda").t().contiguous().t()
    buf = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="must be contiguous"):
        push_buffer_create(inp, buf, locs)

# Rejects input tensor with wrong datatype
# Should raise RuntimeError from TORCH_CHECK macro
def test_rejects_wrong_input_dtype():
    inp = torch.randn(4, 3, dtype=torch.float64, device="cuda")
    buf = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="input datatype must be float32"):
        push_buffer_create(inp, buf, locs)

# Rejects buffer tensor with wrong datatype
# Should raise RuntimeError from TORCH_CHECK macro
def test_rejects_wrong_buffer_dtype():
    inp = torch.randn(4, 3, dtype=torch.float32, device="cuda")
    buf = torch.zeros(2, 3, dtype=torch.float64, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="buffer datatype must be float32"):
        push_buffer_create(inp, buf, locs)

# Rejects locs tensor with wrong datatype
# Should raise RuntimeError from TORCH_CHECK macro
def test_rejects_wrong_locs_dtype():
    inp = torch.randn(4, 3, dtype=torch.float32, device="cpu")
    buf = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="input_locs datatype must be int64"):
        push_buffer_create(inp, buf, locs)

# Rejects 1D input tensor (must be 2D)
# Should raise RuntimeError from TORCH_CHECK macro
def test_rejects_wrong_input_dim():
    inp = torch.randn(12, dtype=torch.float32, device="cuda")
    buf = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="input must have 2 dimensions"):
        push_buffer_create(inp, buf, locs)

# Rejects 1D input tensor (must be 2D)
# Should raise RuntimeError from TORCH_CHECK macro
def test_rejects_wrong_buffer_dim():
    inp = torch.randn(4, 3, dtype=torch.float32, device="cuda")
    buf = torch.zeros(6, dtype=torch.float32, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="buffer must have 2 dimensions"):
        push_buffer_create(inp, buf, locs)

# Rejects 2D input_locs tensor (must be 1D)
# Should raise RuntimeError from TORCH_CHECK macro
def test_rejects_wrong_locs_dim():
    inp = torch.randn(4, 3, dtype=torch.float32, device="cuda")
    buf = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    locs = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="input_locs must have 1 dimension"):
        push_buffer_create(inp, buf, locs)

# Rejects buffer with wrong number of rows (must match input_locs length)
# Should raise RuntimeError from TORCH_CHECK macro
def test_rejects_shape_mismatch_rows():
    inp = torch.randn(4, 3, dtype=torch.float32, device="cuda")
    buf = torch.zeros(5, 3, dtype=torch.float32, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="buffer rows must match size of input_locs"):
        push_buffer_create(inp, buf, locs)

# Rejects buffer with wrong feature dim (must match input feature dim)
# Should raise RuntimeError from TORCH_CHECK macro
def test_rejects_shape_mismatch_features():
    inp = torch.randn(4, 3, dtype=torch.float32, device="cuda")
    buf = torch.zeros(2, 7, dtype=torch.float32, device="cuda")
    locs = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="input and buffer must have same number of columns"):
        push_buffer_create(inp, buf, locs)
