// Scale kernel implementation for sgl-kernel
// Multiplies each element of input tensor by a scalar value

#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel: element-wise scaling
template <typename scalar_t>
__global__ void scale_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const float scale_factor,
    const int64_t numel
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<scalar_t>(
            static_cast<float>(input[idx]) * scale_factor
        );
    }
}

// Host function: validates inputs and launches kernel
torch::Tensor scale_cuda(
    const torch::Tensor& input,
    float scale_factor,
    torch::optional<torch::Tensor> out
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    torch::Tensor output;
    if (out.has_value()) {
        output = out.value();
        TORCH_CHECK(output.is_cuda(), "out must be a CUDA tensor");
        TORCH_CHECK(output.is_contiguous(), "out must be contiguous");
        TORCH_CHECK(output.sizes() == input.sizes(),
                    "out shape must match input shape");
        TORCH_CHECK(output.scalar_type() == input.scalar_type(),
                    "out dtype must match input dtype");
    } else {
        output = torch::empty_like(input);
    }

    const int64_t numel = input.numel();
    if (numel == 0) return output;

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "scale_cuda",
        [&]() {
            scale_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                scale_factor,
                numel
            );
        }
    );

    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "scale",
        &scale_cuda,
        "Element-wise scale (CUDA)",
        py::arg("input"),
        py::arg("scale_factor"),
        py::arg("out") = py::none()
    );
}
