"""Tests for the scale kernel implemented in scale_kernel.cu.

Verifies correctness, edge cases, and performance of the custom
SGL scale kernel against a PyTorch reference implementation.
"""

import pytest
import torch


def torch_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference PyTorch implementation of the scale operation."""
    return x * scale


try:
    from sgl_kernel import scale as sgl_scale
    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


requires_sgl_kernel = pytest.mark.skipif(
    not HAS_SGL_KERNEL, reason="sgl_kernel not installed"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@requires_sgl_kernel
@requires_cuda
class TestScaleCorrectness:
    """Correctness tests comparing sgl_scale to torch_scale."""

    def test_scale_basic(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32)
        scale = 2.5
        expected = torch_scale(x, scale)
        result = sgl_scale(x, scale)
        torch.testing.assert_close(result, expected)

    def test_scale_2d(self):
        x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        scale = 0.5
        expected = torch_scale(x, scale)
        result = sgl_scale(x, scale)
        torch.testing.assert_close(result, expected)

    def test_scale_fp16(self):
        x = torch.randn(256, device="cuda", dtype=torch.float16)
        scale = 3.0
        expected = torch_scale(x, scale)
        result = sgl_scale(x, scale)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_scale_bf16(self):
        x = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        scale = 1.5
        expected = torch_scale(x, scale)
        result = sgl_scale(x, scale)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_scale_zero(self):
        x = torch.randn(64, device="cuda")
        result = sgl_scale(x, 0.0)
        torch.testing.assert_close(result, torch.zeros_like(x))

    def test_scale_negative(self):
        x = torch.randn(64, device="cuda")
        scale = -1.0
        expected = torch_scale(x, scale)
        result = sgl_scale(x, scale)
        torch.testing.assert_close(result, expected)

    def test_scale_large_tensor(self):
        x = torch.randn(1024, 1024, device="cuda")
        scale = 0.125
        expected = torch_scale(x, scale)
        result = sgl_scale(x, scale)
        torch.testing.assert_close(result, expected)

    def test_scale_out_param(self):
        """Test that the out parameter writes results correctly."""
        x = torch.randn(128, device="cuda")
        out = torch.empty_like(x)
        scale = 2.0
        sgl_scale(x, scale, out=out)
        expected = torch_scale(x, scale)
        torch.testing.assert_close(out, expected)


@requires_sgl_kernel
class TestScaleErrorHandling:
    """Tests for expected errors and edge cases."""

    def test_scale_cpu_input_raises(self):
        """CPU tensors should raise an error — kernel requires CUDA."""
        x = torch.randn(64, device="cpu")
        with pytest.raises((RuntimeError, ValueError)):
            sgl_scale(x, 1.0)

    def test_scale_shape_mismatch_raises(self):
        """Mismatched out shape should raise an error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        x = torch.randn(64, device="cuda")
        out = torch.empty(128, device="cuda")
        with pytest.raises((RuntimeError, ValueError)):
            sgl_scale(x, 1.0, out=out)
