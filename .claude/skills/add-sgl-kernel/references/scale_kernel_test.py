"""Tests for the scale kernel implemented in sgl-kernels."""
import pytest
import torch

from sgl_kernel import scale as sgl_scale


def torch_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Reference implementation using pure PyTorch."""
    return x * factor


class TestScaleCorrectness:
    """Correctness tests comparing sgl_scale against torch reference."""

    def test_scale_basic(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32)
        factor = 2.5
        expected = torch_scale(x, factor)
        result = sgl_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_2d(self):
        x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        factor = 0.5
        expected = torch_scale(x, factor)
        result = sgl_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_fp16(self):
        x = torch.randn(256, device="cuda", dtype=torch.float16)
        factor = 3.0
        expected = torch_scale(x, factor)
        result = sgl_scale(x, factor)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_scale_bf16(self):
        x = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        factor = 1.5
        expected = torch_scale(x, factor)
        result = sgl_scale(x, factor)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_scale_negative_factor(self):
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        factor = -1.0
        expected = torch_scale(x, factor)
        result = sgl_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_zero_factor(self):
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        factor = 0.0
        result = sgl_scale(x, factor)
        assert torch.all(result == 0.0)

    def test_scale_large_tensor(self):
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        factor = 0.125
        expected = torch_scale(x, factor)
        result = sgl_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_inplace(self):
        """Verify that the kernel does not modify the input tensor."""
        x = torch.randn(128, device="cuda", dtype=torch.float32)
        x_clone = x.clone()
        factor = 2.0
        _ = sgl_scale(x, factor)
        torch.testing.assert_close(x, x_clone)


class TestScaleEdgeCases:
    def test_scale_cpu_input(self):
        """Kernel should raise when given a CPU tensor."""
        x = torch.randn(64, dtype=torch.float32)  # CPU
        with pytest.raises((RuntimeError, ValueError)):
            sgl_scale(x, 1.0)

    def test_scale_empty_tensor(self):
        x = torch.empty(0, device="cuda", dtype=torch.float32)
        result = sgl_scale(x, 2.0)
        assert result.numel() == 0

    def test_scale_non_contiguous(self):
        """Kernel should handle or reject non-contiguous tensors gracefully."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.float32).t()
        # Either succeeds with correct result or raises a clear error
        try:
            result = sgl_scale(x, 2.0)
            expected = torch_scale(x.contiguous(), 2.0)
            torch.testing.assert_close(result.contiguous(), expected)
        except (RuntimeError, ValueError):
            pass  # Acceptable to reject non-contiguous input
