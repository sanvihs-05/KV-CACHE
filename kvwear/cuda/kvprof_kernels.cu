// Profiling support: empirical roofline ceilings + a static resource/occupancy report.
//
// Ceilings (what this GPU can actually do, measured, not taken from a spec sheet):
//   peak_fp32_fma<CH>  every thread runs CH independent FMA dependency chains, so the FMA
//                      pipes never wait on latency: GFLOP/s = 2 * CH * iters * threads / time
//   stream_read        128-bit grid-stride loads, XOR-reduced, (practically) no stores
//   stream_copy        128-bit load + 128-bit store (what a memcpy-like kernel can reach)
//   stream_write       128-bit stores only
// Resource report: cudaFuncGetAttributes (registers, local = spills/stack, static shared memory)
// and cudaOccupancyMaxActiveBlocksPerMultiprocessor for every kernel in the registry, at the
// block size its launcher really uses.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "kvprof.h"

template <int CH>
__global__ void __launch_bounds__(256) peak_fp32_fma(float* __restrict__ out, int iters, float a, float b) {
    float x[CH];
#pragma unroll
    for (int c = 0; c < CH; ++c) x[c] = threadIdx.x * 1e-3f + (float)c;
#pragma unroll 4
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < CH; ++c) x[c] = __fmaf_rn(x[c], a, b);   // explicit FFMA (fmad=false is irrelevant)
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) s += x[c];
    out[(long)blockIdx.x * blockDim.x + threadIdx.x] = s;              // keeps the chains alive
}

__global__ void stream_read(const uint4* __restrict__ in, uint4* __restrict__ out, long n) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        uint4 v = in[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    // data-dependent store the compiler cannot prove dead; with random input it never happens
    if (acc.x == 0x9e3779b9u && acc.y == 0x7f4a7c15u) out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void stream_copy(const uint4* __restrict__ in, uint4* __restrict__ out, long n) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) out[i] = in[i];
}

__global__ void stream_write(uint4* __restrict__ out, long n, uint32_t v) {
    const uint4 w = make_uint4(v, v ^ 1u, v ^ 2u, v ^ 3u);
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) out[i] = w;
}

// ------------------------------------------------------------------ launchers

#define PEAK_CHAINS 16

void launch_peak_fp32(float* out, int blocks, int iters, cudaStream_t st) {
    peak_fp32_fma<PEAK_CHAINS><<<blocks, 256, 0, st>>>(out, iters, 0.99991f, 1e-4f);
}

int peak_fp32_chains() { return PEAK_CHAINS; }

// mode 0 read, 1 copy, 2 write. n16 = number of 16-byte elements.
void launch_stream(int mode, const void* in, void* out, long n16, int grid, cudaStream_t st) {
    const int T = 256;
    if (mode == 0) stream_read<<<grid, T, 0, st>>>((const uint4*)in, (uint4*)out, n16);
    else if (mode == 1) stream_copy<<<grid, T, 0, st>>>((const uint4*)in, (uint4*)out, n16);
    else stream_write<<<grid, T, 0, st>>>((uint4*)out, n16, 0x5bd1e995u);
}

int kvprof_kernel_refs(KernelRef* out, int max) {
    const KernelRef refs[] = {
        {"peak_fp32_fma<16> (ceiling)", (const void*)peak_fp32_fma<PEAK_CHAINS>, 256},
        {"stream_read (ceiling)", (const void*)stream_read, 256},
        {"stream_copy (ceiling)", (const void*)stream_copy, 256},
        {"stream_write (ceiling)", (const void*)stream_write, 256},
    };
    int n = 0;
    for (const auto& r : refs) if (n < max) out[n++] = r;
    return n;
}

// ------------------------------------------------------------------ static resource report

int kvwear_kernel_attrs(KernelAttr* out, int max) {
    KernelRef refs[160];
    int n = 0;
    n += kvquant_kernel_refs(refs + n, 160 - n);
    n += kvattn_kernel_refs(refs + n, 160 - n);
    n += kvprof_kernel_refs(refs + n, 160 - n);
    int k = 0;
    for (int i = 0; i < n && k < max; ++i) {
        cudaFuncAttributes a;
        if (cudaFuncGetAttributes(&a, refs[i].fn) != cudaSuccess) { cudaGetLastError(); continue; }
        int bps = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, refs[i].fn, refs[i].block, 0) != cudaSuccess) {
            cudaGetLastError();
            bps = -1;
        }
        out[k++] = {refs[i].name, refs[i].block, a.numRegs, (int)a.localSizeBytes, (int)a.sharedSizeBytes,
                    (int)a.constSizeBytes, a.maxThreadsPerBlock, a.ptxVersion, a.binaryVersion, bps};
    }
    return k;
}

// vals: [sms, clock_khz, mem_clock_khz, bus_bits, regs_per_sm, smem_per_sm, smem_per_block_optin,
//        max_threads_per_sm, max_blocks_per_sm, warp_size, l2_bytes, cc_major, cc_minor, reserved_smem_per_block]
void kvwear_device_props(int dev, int* vals) {
    const cudaDeviceAttr attrs[] = {
        cudaDevAttrMultiProcessorCount, cudaDevAttrClockRate, cudaDevAttrMemoryClockRate,
        cudaDevAttrGlobalMemoryBusWidth, cudaDevAttrMaxRegistersPerMultiprocessor,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerBlockOptin,
        cudaDevAttrMaxThreadsPerMultiProcessor, cudaDevAttrMaxBlocksPerMultiprocessor, cudaDevAttrWarpSize,
        cudaDevAttrL2CacheSize, cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
        cudaDevAttrReservedSharedMemoryPerBlock};
    for (int i = 0; i < (int)(sizeof(attrs) / sizeof(attrs[0])); ++i)
        if (cudaDeviceGetAttribute(&vals[i], attrs[i], dev) != cudaSuccess) { vals[i] = -1; cudaGetLastError(); }
}
