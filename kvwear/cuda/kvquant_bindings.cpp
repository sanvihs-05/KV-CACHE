// PyTorch bindings for kvquant_kernels.cu. Every launcher runs on the *current* CUDA
// stream, so callers choose overlap with `with torch.cuda.stream(s):`.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

void launch_quant_int8(const void* x, void* out, long groups, int g, int variant, cudaStream_t st);
void launch_quant_int4(const void* x, void* out, long groups, int g, int variant, cudaStream_t st);
void launch_dequant(const void* payload, void* out, long n, int g, int method, cudaStream_t st);
void launch_dequant_scatter(const void* base, const long* offs, const int* toffs, const int* methods,
                            int nblocks, void* dest, long n, int Tb, int D, int Ttot, int g, cudaStream_t st);
void query_memory_bandwidth(int device, double* peak_gbps, int* clock_khz, int* bus_bits);
void launch_paged_int8_decode(const void* q, const void* base, const long* offs, long groups, int layer,
                              int Hq, int Hkv, int Tb, int T, int D, int g, int tps, float scale,
                              float* pm, float* pl, float* pacc, void* out, int variant, cudaStream_t st);
void launch_dense_fp16_decode(const void* q, const void* K, const void* V, int Hq, int Hkv, int T, int D,
                              int tps, float scale, float* pm, float* pl, float* pacc, void* out, cudaStream_t st);

static void check_half(const torch::Tensor& x, int64_t g) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "input must be float16");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(g > 0 && g % 2 == 0, "group size must be even");
    TORCH_CHECK(x.numel() % g == 0, "numel must be a multiple of the group size");
    TORCH_CHECK(g <= 256, "group size must be <= 256");
}

// Returns the payload (without header): uint8 [2*groups + n]
torch::Tensor quant_int8(torch::Tensor x, int64_t g, int64_t variant) {
    check_half(x, g);
    const c10::cuda::CUDAGuard guard(x.device());
    long n = x.numel(), groups = n / g;
    auto out = torch::empty({2 * groups + n}, x.options().dtype(torch::kUInt8));
    launch_quant_int8(x.data_ptr(), out.data_ptr(), groups, (int)g, (int)variant,
                      c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Returns the payload (without header): uint8 [4*groups + n/2]
torch::Tensor quant_int4(torch::Tensor x, int64_t g, int64_t variant) {
    check_half(x, g);
    const c10::cuda::CUDAGuard guard(x.device());
    long n = x.numel(), groups = n / g;
    auto out = torch::empty({4 * groups + n / 2}, x.options().dtype(torch::kUInt8));
    launch_quant_int4(x.data_ptr(), out.data_ptr(), groups, (int)g, (int)variant,
                      c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// payload uint8 (CUDA) -> float16 [n]
torch::Tensor dequant(torch::Tensor payload, int64_t n, int64_t g, int64_t method) {
    TORCH_CHECK(payload.is_cuda() && payload.scalar_type() == torch::kUInt8 && payload.is_contiguous(),
                "payload must be a contiguous CUDA uint8 tensor");
    TORCH_CHECK(method == 1 || method == 2, "method must be 1 (int8) or 2 (int4)");
    TORCH_CHECK(g > 0 && g % 2 == 0 && n % g == 0, "bad group size");
    long groups = n / g;
    long need = method == 1 ? 2 * groups + n : 4 * groups + n / 2;
    TORCH_CHECK(payload.numel() >= need, "payload too short");
    const c10::cuda::CUDAGuard guard(payload.device());
    auto out = torch::empty({n}, payload.options().dtype(torch::kHalf));
    launch_dequant(payload.data_ptr(), out.data_ptr(), n, (int)g, (int)method, c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Batched: every block b (payload at base[offs[b]], method methods[b]) is decoded into
// dest[:, toffs[b]:toffs[b]+Tb, :], where dest is a contiguous float16 [R, Ttot, D].
void dequant_scatter(torch::Tensor base, torch::Tensor offs, torch::Tensor toffs, torch::Tensor methods,
                     torch::Tensor dest, int64_t Tb, int64_t g) {
    TORCH_CHECK(base.is_cuda() && base.scalar_type() == torch::kUInt8 && base.is_contiguous(), "base: CUDA uint8");
    TORCH_CHECK(offs.is_cuda() && offs.scalar_type() == torch::kLong, "offs: CUDA int64");
    TORCH_CHECK(toffs.is_cuda() && toffs.scalar_type() == torch::kInt, "toffs: CUDA int32");
    TORCH_CHECK(methods.is_cuda() && methods.scalar_type() == torch::kInt, "methods: CUDA int32");
    TORCH_CHECK(dest.is_cuda() && dest.scalar_type() == torch::kHalf && dest.is_contiguous() && dest.dim() == 3,
                "dest: contiguous CUDA float16 [R, Ttot, D]");
    int nb = (int)offs.numel();
    TORCH_CHECK(toffs.numel() == nb && methods.numel() == nb, "metadata length mismatch");
    TORCH_CHECK(nb <= 65535, "too many blocks for one launch");
    long R = dest.size(0), Ttot = dest.size(1), D = dest.size(2);
    TORCH_CHECK(D % 2 == 0 && D % g == 0, "head_dim must be even and a multiple of the group size");
    if (nb == 0) return;
    const c10::cuda::CUDAGuard guard(dest.device());
    launch_dequant_scatter(base.data_ptr(), offs.data_ptr<int64_t>(), toffs.data_ptr<int>(),
                           methods.data_ptr<int>(), nb, dest.data_ptr(), R * Tb * D, (int)Tb, (int)D,
                           (int)Ttot, (int)g, c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void check_q(const torch::Tensor& q, int64_t Hkv) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kHalf && q.is_contiguous() && q.dim() == 2,
                "q: contiguous CUDA float16 [Hq, D]");
    TORCH_CHECK(q.size(1) == 64 || q.size(1) == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(Hkv > 0 && q.size(0) % Hkv == 0, "Hq must be a multiple of Hkv");
}

// Decode attention over int8 KV blocks in kvwear's on-disk format, through a block table.
// base: concatenated block payloads (CUDA uint8), offs[b]: byte offset of block b's payload,
// groups: scale groups per block = layers*2*Hkv*Tb*(D/g). Returns [Hq, D] float16.
torch::Tensor paged_int8_decode(torch::Tensor q, torch::Tensor base, torch::Tensor offs, int64_t groups,
                                int64_t layer, int64_t Hkv, int64_t Tb, int64_t T, int64_t g, int64_t tps,
                                double scale, int64_t variant) {
    check_q(q, Hkv);
    TORCH_CHECK(base.is_cuda() && base.scalar_type() == torch::kUInt8, "base: CUDA uint8");
    TORCH_CHECK(offs.is_cuda() && offs.scalar_type() == torch::kLong, "offs: CUDA int64");
    TORCH_CHECK(T > 0 && T <= offs.numel() * Tb, "T exceeds the block table");
    TORCH_CHECK((2 * groups) % 4 == 0, "payload alignment");
    TORCH_CHECK(g > 0 && q.size(1) % g == 0 && g % (q.size(1) / 32) == 0, "group size must divide head_dim");
    const c10::cuda::CUDAGuard guard(q.device());
    long Hq = q.size(0), D = q.size(1), ns = (T + tps - 1) / tps;
    auto f = q.options().dtype(torch::kFloat);
    auto pm = torch::empty({Hq * ns}, f), pl = torch::empty({Hq * ns}, f), pacc = torch::empty({Hq * ns * D}, f);
    auto out = torch::empty_like(q);
    launch_paged_int8_decode(q.data_ptr(), base.data_ptr(), offs.data_ptr<int64_t>(), groups, (int)layer,
                             (int)Hq, (int)Hkv, (int)Tb, (int)T, (int)D, (int)g, (int)tps, (float)scale,
                             pm.data_ptr<float>(), pl.data_ptr<float>(), pacc.data_ptr<float>(), out.data_ptr(),
                             (int)variant, c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Same algorithm on a dense fp16 cache K, V: [Hkv, T, D] (the like-for-like baseline).
torch::Tensor dense_fp16_decode(torch::Tensor q, torch::Tensor K, torch::Tensor V, int64_t tps, double scale) {
    TORCH_CHECK(K.is_cuda() && K.scalar_type() == torch::kHalf && K.is_contiguous() && K.dim() == 3, "K: [Hkv,T,D] fp16");
    TORCH_CHECK(V.sizes() == K.sizes() && V.is_contiguous() && V.scalar_type() == torch::kHalf, "V must match K");
    check_q(q, K.size(0));
    TORCH_CHECK(K.size(2) == q.size(1), "head_dim mismatch");
    const c10::cuda::CUDAGuard guard(q.device());
    long Hq = q.size(0), D = q.size(1), T = K.size(1), ns = (T + tps - 1) / tps;
    auto f = q.options().dtype(torch::kFloat);
    auto pm = torch::empty({Hq * ns}, f), pl = torch::empty({Hq * ns}, f), pacc = torch::empty({Hq * ns * D}, f);
    auto out = torch::empty_like(q);
    launch_dense_fp16_decode(q.data_ptr(), K.data_ptr(), V.data_ptr(), (int)Hq, (int)K.size(0), (int)T, (int)D,
                             (int)tps, (float)scale, pm.data_ptr<float>(), pl.data_ptr<float>(),
                             pacc.data_ptr<float>(), out.data_ptr(), c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ------------------------------------------------------------------ profiling (kvprof_kernels.cu)
#include "kvprof.h"
int kvwear_kernel_attrs(KernelAttr* out, int max);
void kvwear_device_props(int dev, int* vals);
void launch_peak_fp32(float* out, int blocks, int iters, cudaStream_t st);
int peak_fp32_chains();
void launch_stream(int mode, const void* in, void* out, long n16, int grid, cudaStream_t st);

py::list kernel_attributes() {
    KernelAttr a[160];
    int n = kvwear_kernel_attrs(a, 160);
    py::list out;
    for (int i = 0; i < n; ++i) {
        py::dict d;
        d["name"] = a[i].name; d["block"] = a[i].block; d["regs"] = a[i].regs; d["local_bytes"] = a[i].local_bytes;
        d["static_smem"] = a[i].static_smem; d["const_bytes"] = a[i].const_bytes; d["max_threads"] = a[i].max_threads;
        d["ptx_version"] = a[i].ptx_version; d["binary_version"] = a[i].binary_version;
        d["blocks_per_sm"] = a[i].blocks_per_sm;
        out.append(d);
    }
    return out;
}

py::dict device_props(int64_t device) {
    static const char* keys[] = {"sms", "clock_khz", "mem_clock_khz", "bus_bits", "regs_per_sm", "smem_per_sm",
                                 "smem_per_block_optin", "max_threads_per_sm", "max_blocks_per_sm", "warp_size",
                                 "l2_bytes", "cc_major", "cc_minor", "reserved_smem_per_block"};
    int v[14];
    kvwear_device_props((int)device, v);
    py::dict d;
    for (int i = 0; i < 14; ++i) d[keys[i]] = v[i];
    return d;
}

// FMA-throughput ceiling: returns the per-thread results (so nothing is optimized away)
torch::Tensor peak_fp32(int64_t blocks, int64_t iters) {
    auto out = torch::empty({blocks * 256}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    launch_peak_fp32(out.data_ptr<float>(), (int)blocks, (int)iters, c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// bandwidth ceilings. mode 0 read src, 1 copy src->dst, 2 write dst. Sizes in bytes, multiple of 16.
void stream_bench(int64_t mode, torch::Tensor src, torch::Tensor dst, int64_t grid) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.is_contiguous() && dst.is_contiguous(), "CUDA tensors");
    long nb = (long)(mode == 2 ? dst.nbytes() : src.nbytes());
    TORCH_CHECK(nb % 16 == 0, "size must be a multiple of 16 bytes");
    TORCH_CHECK(mode != 1 || dst.nbytes() >= src.nbytes(), "dst too small");
    TORCH_CHECK(mode != 0 || dst.nbytes() >= (size_t)grid * 256 * 16, "dst must hold grid*256 uint4");
    launch_stream((int)mode, src.data_ptr(), dst.data_ptr(), nb / 16, (int)grid, c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

py::dict device_memory_info(int64_t device) {
    double peak; int clk, bus;
    query_memory_bandwidth((int)device, &peak, &clk, &bus);
    py::dict d;
    d["peak_gbps"] = peak; d["memory_clock_khz"] = clk; d["bus_width_bits"] = bus;
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "kvwear CUDA kernels: KV block quantization and batched dequantize+scatter";
    m.def("quant_int8", &quant_int8, "int8 per-group absmax quantization (payload without header)",
          py::arg("x"), py::arg("g"), py::arg("variant") = 4);
    m.def("quant_int4", &quant_int4, "int4 per-group min/max quantization (payload without header)",
          py::arg("x"), py::arg("g"), py::arg("variant") = 4);
    m.def("dequant", &dequant, "dequantize one payload to float16 [n]",
          py::arg("payload"), py::arg("n"), py::arg("g"), py::arg("method"));
    m.def("dequant_scatter", &dequant_scatter, "batched dequantize + scatter into a [R, Ttot, D] cache");
    m.def("paged_int8_decode", &paged_int8_decode, "decode attention over int8 KV blocks via a block table",
          py::arg("q"), py::arg("base"), py::arg("offs"), py::arg("groups"), py::arg("layer"), py::arg("Hkv"),
          py::arg("Tb"), py::arg("T"), py::arg("g"), py::arg("tps") = 256, py::arg("scale") = 0.125,
          py::arg("variant") = 3);
    m.def("dense_fp16_decode", &dense_fp16_decode, "decode attention over a dense fp16 cache",
          py::arg("q"), py::arg("K"), py::arg("V"), py::arg("tps") = 256, py::arg("scale") = 0.125);
    m.def("device_memory_info", &device_memory_info, "theoretical DRAM bandwidth of a device");
    m.def("kernel_attributes", &kernel_attributes,
          "registers / spills / shared memory / occupancy of every kvwear kernel (cudaFuncGetAttributes)");
    m.def("device_props", &device_props, "SM resources of a device (cudaDeviceGetAttribute)", py::arg("device") = 0);
    m.def("peak_fp32", &peak_fp32, "FMA-throughput ceiling kernel", py::arg("blocks"), py::arg("iters"));
    m.def("peak_fp32_chains", &peak_fp32_chains, "independent FMA chains per thread in peak_fp32");
    m.def("stream_bench", &stream_bench, "bandwidth ceiling kernels: 0 read, 1 copy, 2 write",
          py::arg("mode"), py::arg("src"), py::arg("dst"), py::arg("grid"));
}
