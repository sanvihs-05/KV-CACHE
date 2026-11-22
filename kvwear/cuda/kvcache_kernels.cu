// Append freshly computed K/V rows to a GPU-resident int8 KV cache that uses kvwear's block format
// (one "block" per layer whose token capacity is the cache capacity Tmax):
//   payload = [groups x fp16 scale][n x int8],  row r = (kv * Hkv + h) * Tmax + t,
//   scale index = r * (D/g) + sub-group,        values at r * D.
// The decode-attention kernel (kvattn_kernels.cu, PagedInt8, layer 0, Tb = Tmax) reads it directly.
// Quantization is the codec's: absmax/127 per group of g values, round-half-even, IEEE division.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FULL_MASK 0xffffffffu

// one warp per (K/V, head, new token, sub-group); half2 loads, char2 stores
__global__ void append_int8_rows(const __half* __restrict__ k, const __half* __restrict__ v,
                                 uint8_t* __restrict__ dst, int Hkv, int tnew, int D, int g, int Tmax, int t0,
                                 long groups_total) {
    long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    const int gpr = D / g;
    long work = 2L * Hkv * tnew * gpr;
    if (w >= work) return;
    int sub = (int)(w % gpr);
    long rest = w / gpr;
    int t = (int)(rest % tnew);
    rest /= tnew;
    int h = (int)(rest % Hkv);
    int kv = (int)(rest / Hkv);
    const __half* src = (kv == 0 ? k : v) + ((long)h * tnew + t) * D + (long)sub * g;
    const __half2* src2 = reinterpret_cast<const __half2*>(src);
    const int g2 = g / 2;
    float amax = 0.f;
    for (int i = lane; i < g2; i += 32) {
        float2 f = __half22float2(src2[i]);
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL_MASK, amax, o));
    float s = __fdiv_rn(amax, 127.0f);
    if (s == 0.f) s = 1.f;
    long r = ((long)kv * Hkv + h) * Tmax + t0 + t;
    __half* scales = reinterpret_cast<__half*>(dst);
    if (lane == 0) scales[r * gpr + sub] = __float2half_rn(s);
    char2* q = reinterpret_cast<char2*>(dst + 2 * groups_total + r * D + (long)sub * g);
    for (int i = lane; i < g2; i += 32) {
        float2 f = __half22float2(src2[i]);
        int a = max(-127, min(127, __float2int_rn(rintf(__fdiv_rn(f.x, s)))));
        int b = max(-127, min(127, __float2int_rn(rintf(__fdiv_rn(f.y, s)))));
        q[i] = make_char2((signed char)a, (signed char)b);
    }
}

// dequantize rows [0, T) of one layer back to fp16 K and V: [Hkv, T, D] each (for chunked prefill)
__global__ void gather_int8_rows(const uint8_t* __restrict__ src, __half* __restrict__ k, __half* __restrict__ v,
                                 int Hkv, int T, int D, int g, int Tmax, long groups_total) {
    long n2 = 2L * Hkv * T * D / 2;
    const __half* scales = reinterpret_cast<const __half*>(src);
    const int8_t* q = reinterpret_cast<const int8_t*>(src + 2 * groups_total);
    const int gpr = D / g;
    for (long p = (long)blockIdx.x * blockDim.x + threadIdx.x; p < n2; p += (long)gridDim.x * blockDim.x) {
        long e = 2 * p;                         // element index in [2, Hkv, T, D]
        int d = (int)(e % D);
        long rest = e / D;
        int t = (int)(rest % T);
        rest /= T;
        int h = (int)(rest % Hkv);
        int kv = (int)(rest / Hkv);
        long r = ((long)kv * Hkv + h) * Tmax + t;
        float s = __half2float(scales[r * gpr + d / g]);
        float a = (float)q[r * D + d] * s, b = (float)q[r * D + d + 1] * s;
        __half2* out = reinterpret_cast<__half2*>(kv == 0 ? k : v);
        out[(((long)h * T + t) * D + d) / 2] = __floats2half2_rn(a, b);
    }
}

void launch_append_int8(const void* k, const void* v, void* dst, int Hkv, int tnew, int D, int g, int Tmax, int t0,
                        long groups_total, cudaStream_t st) {
    long warps = 2L * Hkv * tnew * (D / g);
    const int T = 256;
    long blocks = (warps * 32 + T - 1) / T;
    append_int8_rows<<<(unsigned)blocks, T, 0, st>>>((const __half*)k, (const __half*)v, (uint8_t*)dst, Hkv, tnew, D,
                                                     g, Tmax, t0, groups_total);
}

void launch_gather_int8(const void* src, void* k, void* v, int Hkv, int T, int D, int g, int Tmax, long groups_total,
                        cudaStream_t st) {
    long n2 = (long)Hkv * T * D;
    const int TH = 256;
    long blocks = (n2 + TH - 1) / TH;
    if (blocks > 65535L * 8) blocks = 65535L * 8;
    gather_int8_rows<<<(unsigned)blocks, TH, 0, st>>>((const uint8_t*)src, (__half*)k, (__half*)v, Hkv, T, D, g, Tmax,
                                                      groups_total);
}
