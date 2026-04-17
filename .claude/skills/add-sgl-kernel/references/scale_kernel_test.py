"""Tests for scale kernel correctness and edge cases."""
import pytest
import torch
from sgl_kernel import scale as sgl_scale


def torch_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return x * factor


class TestScaleCorrectness:
    """Tests verifying numerical correctness of the scale kernel."""

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

    def test_scale_zero_factor(self):
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        result = sgl_scale(x, 0.0)
        torch.testing.assert_close(result, torch.zeros_like(x))

    def test_scale_negative_factor(self):
        x = torch.randn(64, device="cuda", dtype=torch.float32)
        factor = -1.0
        expected = torch_scale(x, factor)
        result = sgl_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_large_tensor(self):
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        factor = 0.1
        expected = torch_scale(x, factor)
        result = sgl_scale(x, factor)
        torch.testing.assert_close(result, expected)

    def test_scale_inplace(self):
        x = torch.randn(128, device="cuda", dtype=torch.float32)
        x_clone = x.clone()
        factor = 2.0
        sgl_scale(x, factor, inplace=True)
        torch.testing.assert_close(x, x_clone * factor)

    def test_scale_cpu_error(self):
        x = torch.randn(64, dtype=torch.float32)  # CPU tensor
        with pytest.raises(RuntimeError, match="CUDA"):
            sgl_scale(x, 1.0)

    def test_scale_non_contiguous(self):
        x = torch.randn(128, 64, device="cuda", dtype=torch.float32)
        x_strided = x[::2, :]  # non-contiguous view
        factor = 2.0
        expected = torch_scale(x_strided.contiguous(), factor)
        result = sgl_scale(x_strided, factor)
        torch.testing.assert_close(result, expected)
