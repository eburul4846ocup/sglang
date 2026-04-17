"""Tests for the scale kernel implementation.

Verifies correctness, edge cases, and error handling for the
custom CUDA scale kernel via the sgl-kernel Python bindings.
"""

import pytest
import torch
from sgl_kernel import scale as sgl_scale


def torch_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return x * factor


class TestScaleCorrectness:
    """Correctness tests comparing sgl_scale against torch reference."""

    def test_scale_basic(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32)
        factor = 2.5
        result = sgl_scale(x, factor)
        expected = torch_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_2d(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
        factor = 0.5
        result = sgl_scale(x, factor)
        expected = torch_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_fp16(self):
        x = torch.randn(256, device="cuda", dtype=torch.float16)
        factor = 3.0
        result = sgl_scale(x, factor)
        expected = torch_scale(x, factor)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_scale_bf16(self):
        x = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        factor = 1.5
        result = sgl_scale(x, factor)
        expected = torch_scale(x, factor)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_scale_negative_factor(self):
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        factor = -1.0
        result = sgl_scale(x, factor)
        expected = torch_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_zero_factor(self):
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        factor = 0.0
        result = sgl_scale(x, factor)
        expected = torch.zeros_like(x)
        torch.testing.assert_close(result, expected)

    def test_scale_large_tensor(self):
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        factor = 0.1
        result = sgl_scale(x, factor)
        expected = torch_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_out_param(self):
        """Test writing result into a pre-allocated output tensor."""
        x = torch.randn(128, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        factor = 2.0
        sgl_scale(x, factor, out=out)
        expected = torch_scale(x, factor)
        torch.testing.assert_close(out, expected)


class TestScaleErrors:
    """Error handling and input validation tests."""

    def test_scale_cpu_input(self):
        """Kernel should raise when given a CPU tensor."""
        x = torch.randn(64, dtype=torch.float32)  # CPU
        with pytest.raises((RuntimeError, ValueError)):
            sgl_scale(x, 1.0)

    def test_scale_shape_mismatch(self):
        """Output tensor shape must match input shape."""
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        out = torch.empty(128, device="cuda", dtype=torch.float32)
        with pytest.raises((RuntimeError, ValueError)):
            sgl_scale(x, 1.0, out=out)

    def test_scale_dtype_mismatch(self):
        """Output tensor dtype must match input dtype."""
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        out = torch.empty(64, device="cuda", dtype=torch.float16)
        with pytest.raises((RuntimeError, ValueError)):
            sgl_scale(x, 1.0, out=out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
