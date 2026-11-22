// Bindings for kvcache_kernels.cu (GPU-resident int8 KV cache used during generation).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

void launch_append_int8(const void* k, const void* v, void* dst, int Hkv, int tnew, int D, int g, int Tmax, int t0,
                        long groups_total, cudaStream_t st);
void launch_gather_int8(const void* src, void* k, void* v, int Hkv, int T, int D, int g, int Tmax, long groups_total,
                        cudaStream_t st);

// k, v: [Hkv, tnew, D] fp16 contiguous. dst: the layer's uint8 buffer.
void append_int8(torch::Tensor k, torch::Tensor v, torch::Tensor dst, int64_t Tmax, int64_t t0, int64_t g) {
    TORCH_CHECK(k.is_cuda() && k.scalar_type() == torch::kHalf && k.is_contiguous() && k.dim() == 3, "k: [Hkv,t,D] fp16");
    TORCH_CHECK(v.sizes() == k.sizes() && v.is_contiguous() && v.scalar_type() == torch::kHalf, "v must match k");
    TORCH_CHECK(dst.is_cuda() && dst.scalar_type() == torch::kUInt8 && dst.is_contiguous(), "dst: CUDA uint8");
    long Hkv = k.size(0), tnew = k.size(1), D = k.size(2);
    TORCH_CHECK(D % g == 0 && g % 2 == 0, "group must divide head_dim");
    TORCH_CHECK(t0 + tnew <= Tmax, "int8 cache capacity exceeded");
    long groups_total = 2 * Hkv * Tmax * (D / g);
    TORCH_CHECK(dst.numel() >= 2 * groups_total + 2 * Hkv * Tmax * D, "dst too small");
    if (tnew == 0) return;
    const c10::cuda::CUDAGuard guard(k.device());
    launch_append_int8(k.data_ptr(), v.data_ptr(), dst.data_ptr(), (int)Hkv, (int)tnew, (int)D, (int)g, (int)Tmax,
                       (int)t0, groups_total, c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> gather_int8(torch::Tensor src, int64_t Hkv, int64_t T, int64_t D, int64_t Tmax, int64_t g) {
    TORCH_CHECK(src.is_cuda() && src.scalar_type() == torch::kUInt8, "src: CUDA uint8");
    const c10::cuda::CUDAGuard guard(src.device());
    auto opts = src.options().dtype(torch::kHalf);
    auto k = torch::empty({Hkv, T, D}, opts), v = torch::empty({Hkv, T, D}, opts);
    if (T > 0)
        launch_gather_int8(src.data_ptr(), k.data_ptr(), v.data_ptr(), (int)Hkv, (int)T, (int)D, (int)g, (int)Tmax,
                           2 * Hkv * Tmax * (D / g), c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {k, v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("append_int8", &append_int8, "quantize new K/V rows into a GPU int8 cache layer");
    m.def("gather_int8", &gather_int8, "dequantize the first T tokens of a GPU int8 cache layer");
}
