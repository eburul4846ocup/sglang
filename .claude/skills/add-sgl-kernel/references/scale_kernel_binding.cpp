// PyTorch C++ binding for the scale kernel.
// Exposes the CUDA scale kernel to Python via torch extension.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declaration from scale_kernel.cu
void launch_scale_kernel(
    const float* input,
    float* output,
    float scale,
    int64_t numel,
    cudaStream_t stream
);

/// Apply element-wise scaling: output = input * scale
///
/// Args:
///   input: Input tensor (CUDA, float32)
///   scale: Scalar multiplier
///   output: Optional pre-allocated output tensor. If None, a new tensor
///           is allocated with the same shape as input.
///
/// Returns:
///   Scaled output tensor (same shape and device as input)
torch::Tensor scale(
    const torch::Tensor& input,
    float scale_factor,
    c10::optional<torch::Tensor> output
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "Input must be float32, got ", input.scalar_type());
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    torch::Tensor out;
    if (output.has_value()) {
        out = output.value();
        TORCH_CHECK(out.is_cuda(), "Output must be a CUDA tensor");
        TORCH_CHECK(out.scalar_type() == torch::kFloat32,
                    "Output must be float32");
        TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");
        TORCH_CHECK(out.sizes() == input.sizes(),
                    "Output shape ", out.sizes(),
                    " must match input shape ", input.sizes());
    } else {
        out = torch::empty_like(input);
    }

    int64_t numel = input.numel();
    if (numel == 0) {
        return out;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_scale_kernel(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        scale_factor,
        numel,
        stream
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SGLang scale kernel: element-wise scalar multiplication";
    m.def(
        "scale",
        &scale,
        py::arg("input"),
        py::arg("scale"),
        py::arg("output") = py::none(),
        R"doc(
            Apply element-wise scaling to a CUDA float32 tensor.

            Args:
                input (torch.Tensor): Input tensor on CUDA device.
                scale (float): Scalar multiplier.
                output (torch.Tensor, optional): Pre-allocated output buffer.

            Returns:
                torch.Tensor: Tensor with values multiplied by scale.
        )doc"
    );
}
