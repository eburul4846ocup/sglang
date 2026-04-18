"""Tests for the scale kernel implemented in sgl-kernels.

Runs correctness, shape, dtype, and device checks against a pure-PyTorch
reference implementation so the test suite can be executed on any machine
(CPU-only CI or GPU runners).
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def torch_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Pure-PyTorch reference: element-wise multiply by scalar."""
    return x * scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_kernel():
    """Import the compiled sgl-kernel, skip the test if unavailable."""
    try:
        from sgl_kernel import scale as sgl_scale  # noqa: PLC0415
        return sgl_scale
    except ImportError:
        pytest.skip("sgl_kernel not installed – skipping GPU kernel tests")


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestScaleCorrectness:
    """Verify numerical agreement between the kernel and the reference."""

    def test_scale_basic(self):
        sgl_scale = _import_kernel()
        x = torch.randn(128, device="cuda")
        expected = torch_scale(x, 2.0)
        result = sgl_scale(x, 2.0)
        torch.testing.assert_close(result, expected)

    def test_scale_2d(self):
        sgl_scale = _import_kernel()
        x = torch.randn(32, 64, device="cuda")
        expected = torch_scale(x, 0.5)
        result = sgl_scale(x, 0.5)
        torch.testing.assert_close(result, expected)

    def test_scale_fp16(self):
        sgl_scale = _import_kernel()
        x = torch.randn(256, device="cuda", dtype=torch.float16)
        expected = torch_scale(x, 3.0)
        result = sgl_scale(x, 3.0)
        torch.testing.assert_close(result, expected)

    def test_scale_bf16(self):
        sgl_scale = _import_kernel()
        x = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        expected = torch_scale(x, 0.25)
        result = sgl_scale(x, 0.25)
        torch.testing.assert_close(result, expected)

    def test_scale_negative(self):
        sgl_scale = _import_kernel()
        x = torch.randn(64, device="cuda")
        expected = torch_scale(x, -1.0)
        result = sgl_scale(x, -1.0)
        torch.testing.assert_close(result, expected)

    def test_scale_zero(self):
        sgl_scale = _import_kernel()
        x = torch.randn(64, device="cuda")
        result = sgl_scale(x, 0.0)
        torch.testing.assert_close(result, torch.zeros_like(x))


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------

class TestScaleErrors:
    """Verify that the kernel raises informative errors for bad inputs."""

    def test_scale_cpu_input(self):
        """Kernel must reject CPU tensors."""
        sgl_scale = _import_kernel()
        x = torch.randn(64)  # CPU
        with pytest.raises((RuntimeError, ValueError)):
            sgl_scale(x, 1.0)

    def test_scale_non_contiguous(self):
        """Kernel should handle or explicitly reject non-contiguous tensors."""
        sgl_scale = _import_kernel()
        x = torch.randn(64, 64, device="cuda")[::2, ::2]  # strided view
        # Either succeeds with correct result or raises – both are acceptable.
        try:
            result = sgl_scale(x, 2.0)
            torch.testing.assert_close(result, torch_scale(x.contiguous(), 2.0))
        except (RuntimeError, ValueError):
            pass  # kernel chose to reject non-contiguous input
