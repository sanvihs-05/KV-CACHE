// kvwear CUDA kernels: KV-cache block quantization (int8 / int4) and a batched
// "dequantize + scatter into the attention cache" kernel.
//
// Byte format (identical to the CPU codec in kvwear/codecs.py, enforced by tests):
//   int8 : [groups x fp16 scale][n x int8]                     scale = absmax/127
//   int4 : [groups x fp16 scale][groups x fp16 min][n/2 bytes]  scale = (max-min)/15,
//          byte k = q[2k] | q[2k+1] << 4
// A "group" is `g` consecutive values along head_dim. Quantization uses the fp32 scale;
// only the stored copy is rounded to fp16. Rounding is round-half-to-even (rintf), and
// FMA contraction is disabled with __fmul_rn/__fadd_rn so results match the CPU bit for bit.
//
// Quantize kernels come in four optimization stages (selected by `variant`):
//   0 naive      one thread per group, serial loop      (uncoalesced: lanes are g*2 B apart)
//   1 warp       one warp per group, scalar half loads, __shfl_xor_sync reduction
//   2 warp_vec   one warp per group, half2 (4-byte) vector loads/stores
//   3 warp_reg   like 2, but the group is kept in registers: global memory is read once
//   4 subwarp    g/8 lanes per group (a warp quantizes 32/(g/8) groups at once), 128-bit loads
//                (8 halves per lane), 8-byte int8 / 4-byte int4 stores, shuffles inside the
//                sub-warp only. Fewer, wider memory transactions -> close to memcpy bandwidth.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FULL_MASK 0xffffffffu
#define MAX_REG_PAIRS 4            // variant 3 caches up to 4 half2 per lane: g <= 256

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}
__device__ __forceinline__ float warp_min(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}
__device__ __forceinline__ int q8(float x, float s) {
    int q = __float2int_rn(rintf(__fdiv_rn(x, s)));
    return max(-127, min(127, q));
}
__device__ __forceinline__ int q4(float x, float mn, float s) {
    int q = __float2int_rn(rintf(__fdiv_rn(__fsub_rn(x, mn), s)));
    return max(0, min(15, q));
}

// ------------------------------------------------------------------ int8 quantize

__global__ void quant_int8_naive(const __half* __restrict__ x, __half* __restrict__ scales,
                                 int8_t* __restrict__ q, long groups, int g) {
    long grp = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (grp >= groups) return;
    const __half* row = x + grp * g;
    float amax = 0.f;
    for (int i = 0; i < g; ++i) amax = fmaxf(amax, fabsf(__half2float(row[i])));
    float s = __fdiv_rn(amax, 127.0f);
    if (s == 0.f) s = 1.f;
    scales[grp] = __float2half_rn(s);
    for (int i = 0; i < g; ++i) q[grp * g + i] = (int8_t)q8(__half2float(row[i]), s);
}

__global__ void quant_int8_warp(const __half* __restrict__ x, __half* __restrict__ scales,
                                int8_t* __restrict__ q, long groups, int g) {
    long grp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (grp >= groups) return;                      // whole warp exits together
    const __half* row = x + grp * g;
    float amax = 0.f;
    for (int i = lane; i < g; i += 32) amax = fmaxf(amax, fabsf(__half2float(row[i])));
    amax = warp_max(amax);
    float s = __fdiv_rn(amax, 127.0f);
    if (s == 0.f) s = 1.f;
    if (lane == 0) scales[grp] = __float2half_rn(s);
    for (int i = lane; i < g; i += 32) q[grp * g + i] = (int8_t)q8(__half2float(row[i]), s);
}

__global__ void quant_int8_warp_vec(const __half2* __restrict__ x, __half* __restrict__ scales,
                                    char2* __restrict__ q, long groups, int g2) {
    long grp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (grp >= groups) return;
    const __half2* row = x + grp * g2;
    float amax = 0.f;
    for (int i = lane; i < g2; i += 32) {
        float2 v = __half22float2(row[i]);
        amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    amax = warp_max(amax);
    float s = __fdiv_rn(amax, 127.0f);
    if (s == 0.f) s = 1.f;
    if (lane == 0) scales[grp] = __float2half_rn(s);
    for (int i = lane; i < g2; i += 32) {
        float2 v = __half22float2(row[i]);
        q[grp * g2 + i] = make_char2((signed char)q8(v.x, s), (signed char)q8(v.y, s));
    }
}

__global__ void quant_int8_warp_reg(const __half2* __restrict__ x, __half* __restrict__ scales,
                                    char2* __restrict__ q, long groups, int g2) {
    long grp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (grp >= groups) return;
    const __half2* row = x + grp * g2;
    float2 v[MAX_REG_PAIRS];
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < MAX_REG_PAIRS; ++k) {
        int i = lane + 32 * k;
        v[k] = (i < g2) ? __half22float2(row[i]) : make_float2(0.f, 0.f);
        amax = fmaxf(amax, fmaxf(fabsf(v[k].x), fabsf(v[k].y)));
    }
    amax = warp_max(amax);
    float s = __fdiv_rn(amax, 127.0f);
    if (s == 0.f) s = 1.f;
    if (lane == 0) scales[grp] = __float2half_rn(s);
#pragma unroll
    for (int k = 0; k < MAX_REG_PAIRS; ++k) {
        int i = lane + 32 * k;
        if (i < g2) q[grp * g2 + i] = make_char2((signed char)q8(v[k].x, s), (signed char)q8(v[k].y, s));
    }
}

// ------------------------------------------------------------------ int4 quantize

__global__ void quant_int4_naive(const __half* __restrict__ x, __half* __restrict__ scales,
                                 __half* __restrict__ mins, uint8_t* __restrict__ packed,
                                 long groups, int g) {
    long grp = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (grp >= groups) return;
    const __half* row = x + grp * g;
    float mn = INFINITY, mx = -INFINITY;
    for (int i = 0; i < g; ++i) { float f = __half2float(row[i]); mn = fminf(mn, f); mx = fmaxf(mx, f); }
    float s = __fdiv_rn(__fsub_rn(mx, mn), 15.0f);
    if (s == 0.f) s = 1.f;
    scales[grp] = __float2half_rn(s);
    mins[grp] = __float2half_rn(mn);
    for (int i = 0; i < g; i += 2) {
        int a = q4(__half2float(row[i]), mn, s), b = q4(__half2float(row[i + 1]), mn, s);
        packed[(grp * g + i) >> 1] = (uint8_t)(a | (b << 4));
    }
}

__global__ void quant_int4_warp(const __half* __restrict__ x, __half* __restrict__ scales,
                                __half* __restrict__ mins, uint8_t* __restrict__ packed,
                                long groups, int g) {
    long grp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (grp >= groups) return;
    const __half* row = x + grp * g;
    float mn = INFINITY, mx = -INFINITY;
    for (int i = lane; i < g; i += 32) { float f = __half2float(row[i]); mn = fminf(mn, f); mx = fmaxf(mx, f); }
    mn = warp_min(mn);
    mx = warp_max(mx);
    float s = __fdiv_rn(__fsub_rn(mx, mn), 15.0f);
    if (s == 0.f) s = 1.f;
    if (lane == 0) { scales[grp] = __float2half_rn(s); mins[grp] = __float2half_rn(mn); }
    // scalar loads: each lane packs one pair (i, i+1)
    for (int i = 2 * lane; i < g; i += 64) {
        int a = q4(__half2float(row[i]), mn, s), b = q4(__half2float(row[i + 1]), mn, s);
        packed[(grp * g + i) >> 1] = (uint8_t)(a | (b << 4));
    }
}

__global__ void quant_int4_warp_vec(const __half2* __restrict__ x, __half* __restrict__ scales,
                                    __half* __restrict__ mins, uint8_t* __restrict__ packed,
                                    long groups, int g2) {
    long grp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (grp >= groups) return;
    const __half2* row = x + grp * g2;
    float mn = INFINITY, mx = -INFINITY;
    for (int i = lane; i < g2; i += 32) {
        float2 v = __half22float2(row[i]);
        mn = fminf(mn, fminf(v.x, v.y));
        mx = fmaxf(mx, fmaxf(v.x, v.y));
    }
    mn = warp_min(mn);
    mx = warp_max(mx);
    float s = __fdiv_rn(__fsub_rn(mx, mn), 15.0f);
    if (s == 0.f) s = 1.f;
    if (lane == 0) { scales[grp] = __float2half_rn(s); mins[grp] = __float2half_rn(mn); }
    for (int i = lane; i < g2; i += 32) {
        float2 v = __half22float2(row[i]);
        packed[grp * g2 + i] = (uint8_t)(q4(v.x, mn, s) | (q4(v.y, mn, s) << 4));
    }
}

__global__ void quant_int4_warp_reg(const __half2* __restrict__ x, __half* __restrict__ scales,
                                    __half* __restrict__ mins, uint8_t* __restrict__ packed,
                                    long groups, int g2) {
    long grp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (grp >= groups) return;
    const __half2* row = x + grp * g2;
    float2 v[MAX_REG_PAIRS];
    float mn = INFINITY, mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < MAX_REG_PAIRS; ++k) {
        int i = lane + 32 * k;
        if (i < g2) {
            v[k] = __half22float2(row[i]);
            mn = fminf(mn, fminf(v[k].x, v[k].y));
            mx = fmaxf(mx, fmaxf(v[k].x, v[k].y));
        }
    }
    mn = warp_min(mn);
    mx = warp_max(mx);
    float s = __fdiv_rn(__fsub_rn(mx, mn), 15.0f);
    if (s == 0.f) s = 1.f;
    if (lane == 0) { scales[grp] = __float2half_rn(s); mins[grp] = __float2half_rn(mn); }
#pragma unroll
    for (int k = 0; k < MAX_REG_PAIRS; ++k) {
        int i = lane + 32 * k;
        if (i < g2) packed[grp * g2 + i] = (uint8_t)(q4(v[k].x, mn, s) | (q4(v[k].y, mn, s) << 4));
    }
}


// ------------------------------------------------------------------ v4: sub-warp groups, 128-bit I/O

template <int LPG>
__device__ __forceinline__ float sub_max(float v) {
#pragma unroll
    for (int o = LPG / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}
template <int LPG>
__device__ __forceinline__ float sub_min(float v) {
#pragma unroll
    for (int o = LPG / 2; o > 0; o >>= 1) v = fminf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}
__device__ __forceinline__ void load8(const uint4* p, bool ok, float (&f)[8]) {
    uint4 r = ok ? *p : make_uint4(0, 0, 0, 0);
    const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
    for (int k = 0; k < 4; ++k) { float2 t = __half22float2(h[k]); f[2 * k] = t.x; f[2 * k + 1] = t.y; }
}

template <int LPG>   // lanes per group = g / 8
__global__ void quant_int8_subwarp(const uint4* __restrict__ x, __half* __restrict__ scales,
                                   uint2* __restrict__ q, long groups) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long grp = tid / LPG;
    int sl = (int)(tid % LPG);
    bool ok = grp < groups;                    // no early return: every lane joins the shuffles
    float f[8];
    load8(x + tid, ok, f);
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) amax = fmaxf(amax, fabsf(f[k]));
    amax = sub_max<LPG>(amax);
    float s = __fdiv_rn(amax, 127.0f);
    if (s == 0.f) s = 1.f;
    if (!ok) return;
    if (sl == 0) scales[grp] = __float2half_rn(s);
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int k = 0; k < 8; ++k) w[k >> 2] |= (uint32_t)(uint8_t)(int8_t)q8(f[k], s) << (8 * (k & 3));
    q[tid] = make_uint2(w[0], w[1]);
}

template <int LPG>
__global__ void quant_int4_subwarp(const uint4* __restrict__ x, __half* __restrict__ scales,
                                   __half* __restrict__ mins, uint32_t* __restrict__ packed, long groups) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long grp = tid / LPG;
    int sl = (int)(tid % LPG);
    bool ok = grp < groups;
    float f[8];
    load8(x + tid, ok, f);
    float mn = INFINITY, mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < 8; ++k) { mn = fminf(mn, f[k]); mx = fmaxf(mx, f[k]); }
    mn = sub_min<LPG>(mn);
    mx = sub_max<LPG>(mx);
    float s = __fdiv_rn(__fsub_rn(mx, mn), 15.0f);
    if (s == 0.f) s = 1.f;
    if (!ok) return;
    if (sl == 0) { scales[grp] = __float2half_rn(s); mins[grp] = __float2half_rn(mn); }
    uint32_t w = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) w |= (uint32_t)q4(f[k], mn, s) << (4 * k);
    packed[tid] = w;
}

static bool v4_ok(const void* x, const void* q, long groups, int g, int qalign) {
    return g % 8 == 0 && (g / 8) <= 32 && ((g / 8) & (g / 8 - 1)) == 0 &&
           ((uintptr_t)x % 16 == 0) && ((uintptr_t)q % qalign == 0);
}

#define DISPATCH_LPG(LPG_VAL, KERNEL, ...)                                   \
    switch (LPG_VAL) {                                                       \
    case 1: KERNEL<1><<<grid, T, 0, st>>>(__VA_ARGS__); break;               \
    case 2: KERNEL<2><<<grid, T, 0, st>>>(__VA_ARGS__); break;               \
    case 4: KERNEL<4><<<grid, T, 0, st>>>(__VA_ARGS__); break;               \
    case 8: KERNEL<8><<<grid, T, 0, st>>>(__VA_ARGS__); break;               \
    case 16: KERNEL<16><<<grid, T, 0, st>>>(__VA_ARGS__); break;             \
    default: KERNEL<32><<<grid, T, 0, st>>>(__VA_ARGS__); break;             \
    }

// ------------------------------------------------------------------ dequantize (one block, contiguous)

__global__ void dequant_int8(const uint8_t* __restrict__ payload, __half2* __restrict__ out,
                             long groups, long n2, int g) {
    const __half* scales = reinterpret_cast<const __half*>(payload);
    const char2* q = reinterpret_cast<const char2*>(payload + 2 * groups);
    for (long p = (long)blockIdx.x * blockDim.x + threadIdx.x; p < n2; p += (long)gridDim.x * blockDim.x) {
        float s = __half2float(scales[(2 * p) / g]);
        char2 c = q[p];
        out[p] = __floats2half2_rn(__fmul_rn((float)c.x, s), __fmul_rn((float)c.y, s));
    }
}

__global__ void dequant_int4(const uint8_t* __restrict__ payload, __half2* __restrict__ out,
                             long groups, long n2, int g) {
    const __half* scales = reinterpret_cast<const __half*>(payload);
    const __half* mins = scales + groups;
    const uint8_t* packed = payload + 4 * groups;
    for (long p = (long)blockIdx.x * blockDim.x + threadIdx.x; p < n2; p += (long)gridDim.x * blockDim.x) {
        long grp = (2 * p) / g;
        float s = __half2float(scales[grp]), mn = __half2float(mins[grp]);
        uint8_t b = packed[p];
        out[p] = __floats2half2_rn(__fadd_rn(__fmul_rn((float)(b & 0x0F), s), mn),
                                   __fadd_rn(__fmul_rn((float)(b >> 4), s), mn));
    }
}

// ------------------------------------------------------------------ batched dequantize + scatter
// Assembles the attention cache for one request in a single launch.
// All blocks have the same logical shape [R, Tb, D] (R = layers*2*kv_heads) and are written
// into dest [R, Ttot, D] at token offset toff[b]. blockIdx.y selects the KV block, so the
// method branch is uniform per CUDA block (no warp divergence). method 0 = fp16 copy.

__global__ void dequant_scatter(const uint8_t* __restrict__ base, const long* __restrict__ offs,
                                const int* __restrict__ toffs, const int* __restrict__ methods,
                                __half2* __restrict__ dest, long n2, long groups, int Tb, int D,
                                int Ttot, int g) {
    const int b = blockIdx.y;
    const uint8_t* payload = base + offs[b];
    const int m = methods[b];
    const int toff = toffs[b];
    const int D2 = D >> 1;
    for (long p = (long)blockIdx.x * blockDim.x + threadIdx.x; p < n2; p += (long)gridDim.x * blockDim.x) {
        long row = p / D2;                   // row over [R, Tb]
        int col2 = (int)(p - row * D2);
        long drow = (row / Tb) * Ttot + toff + (row % Tb);
        __half2 v;
        if (m == 0) {
            v = reinterpret_cast<const __half2*>(payload)[p];
        } else if (m == 1) {
            const __half* scales = reinterpret_cast<const __half*>(payload);
            char2 c = reinterpret_cast<const char2*>(payload + 2 * groups)[p];
            float s = __half2float(scales[(2 * p) / g]);
            v = __floats2half2_rn(__fmul_rn((float)c.x, s), __fmul_rn((float)c.y, s));
        } else {
            const __half* scales = reinterpret_cast<const __half*>(payload);
            long grp = (2 * p) / g;
            float s = __half2float(scales[grp]), mn = __half2float(scales[groups + grp]);
            uint8_t bt = (payload + 4 * groups)[p];
            v = __floats2half2_rn(__fadd_rn(__fmul_rn((float)(bt & 0x0F), s), mn),
                                  __fadd_rn(__fmul_rn((float)(bt >> 4), s), mn));
        }
        dest[drow * D2 + col2] = v;
    }
}


// ------------------------------------------------------------------ vectorized batched dequantize + scatter
// 8 values per thread: one 8-byte int8 / 4-byte int4 / 16-byte fp16 load, one 16-byte store.
// Needs D % 8 == 0, g % 8 == 0 (the 8 values share a scale) and 8-byte-aligned int8 payloads.
__device__ __forceinline__ uint4 decode8(const uint8_t* __restrict__ payload, int m, long groups, int g, long p8) {
    uint4 out;
    __half2* o = reinterpret_cast<__half2*>(&out);
    if (m == 0) {
        out = reinterpret_cast<const uint4*>(payload)[p8];
    } else if (m == 1) {
        const float s = __half2float(reinterpret_cast<const __half*>(payload)[(8 * p8) / g]);
        uint2 raw = reinterpret_cast<const uint2*>(payload + 2 * groups)[p8];
        const int8_t* q = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            o[k] = __floats2half2_rn(__fmul_rn((float)q[2 * k], s), __fmul_rn((float)q[2 * k + 1], s));
    } else {
        const long grp = (8 * p8) / g;
        const __half* sc = reinterpret_cast<const __half*>(payload);
        const float s = __half2float(sc[grp]), mn = __half2float(sc[groups + grp]);
        uint32_t w = reinterpret_cast<const uint32_t*>(payload + 4 * groups)[p8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float a = (float)((w >> (8 * k)) & 0x0F), b = (float)((w >> (8 * k + 4)) & 0x0F);
            o[k] = __floats2half2_rn(__fadd_rn(__fmul_rn(a, s), mn), __fadd_rn(__fmul_rn(b, s), mn));
        }
    }
    return out;
}

__global__ void dequant_scatter_v8(const uint8_t* __restrict__ base, const long* __restrict__ offs,
                                   const int* __restrict__ toffs, const int* __restrict__ methods,
                                   uint4* __restrict__ dest, long n8, long groups, int Tb, int D,
                                   int Ttot, int g) {
    const int b = blockIdx.y;
    const uint8_t* payload = base + offs[b];
    const int m = methods[b];
    const int toff = toffs[b];
    const int D8 = D >> 3;
    for (long p = (long)blockIdx.x * blockDim.x + threadIdx.x; p < n8; p += (long)gridDim.x * blockDim.x) {
        long row = p / D8;
        int col8 = (int)(p - row * D8);
        long drow = (row / Tb) * Ttot + toff + (row % Tb);
        dest[drow * D8 + col8] = decode8(payload, m, groups, g, p);
    }
}

__global__ void dequant_v8(const uint8_t* __restrict__ payload, uint4* __restrict__ out, long groups, long n8,
                           int g, int m) {
    for (long p = (long)blockIdx.x * blockDim.x + threadIdx.x; p < n8; p += (long)gridDim.x * blockDim.x)
        out[p] = decode8(payload, m, groups, g, p);
}

// ------------------------------------------------------------------ host launchers

static inline int ceil_div_l(long a, long b) { return (int)((a + b - 1) / b); }
static inline int grid_for(long work, int threads, int max_blocks = 65535 * 8) {
    long b = (work + threads - 1) / threads;
    return (int)(b < 1 ? 1 : (b > max_blocks ? max_blocks : b));
}

void launch_quant_int8(const void* x, void* out, long groups, int g, int variant, cudaStream_t st) {
    __half* scales = reinterpret_cast<__half*>(out);
    int8_t* q = reinterpret_cast<int8_t*>(reinterpret_cast<uint8_t*>(out) + 2 * groups);
    const int T = 256;
    if (variant >= 4 && v4_ok(x, q, groups, g, 8)) {
        long threads = groups * (g / 8);
        int grid = ceil_div_l(threads, T);
        DISPATCH_LPG(g / 8, quant_int8_subwarp, (const uint4*)x, scales, (uint2*)q, groups);
        return;
    }
    if (variant >= 4) variant = 3;            // shape/alignment not eligible: fall back
    switch (variant) {
    case 0: quant_int8_naive<<<ceil_div_l(groups, T), T, 0, st>>>((const __half*)x, scales, q, groups, g); break;
    case 1: quant_int8_warp<<<ceil_div_l(groups * 32, T), T, 0, st>>>((const __half*)x, scales, q, groups, g); break;
    case 2: quant_int8_warp_vec<<<ceil_div_l(groups * 32, T), T, 0, st>>>((const __half2*)x, scales, (char2*)q, groups, g / 2); break;
    default: quant_int8_warp_reg<<<ceil_div_l(groups * 32, T), T, 0, st>>>((const __half2*)x, scales, (char2*)q, groups, g / 2); break;
    }
}

void launch_quant_int4(const void* x, void* out, long groups, int g, int variant, cudaStream_t st) {
    __half* scales = reinterpret_cast<__half*>(out);
    __half* mins = scales + groups;
    uint8_t* packed = reinterpret_cast<uint8_t*>(out) + 4 * groups;
    const int T = 256;
    if (variant >= 4 && v4_ok(x, packed, groups, g, 4)) {
        long threads = groups * (g / 8);
        int grid = ceil_div_l(threads, T);
        DISPATCH_LPG(g / 8, quant_int4_subwarp, (const uint4*)x, scales, mins, (uint32_t*)packed, groups);
        return;
    }
    if (variant >= 4) variant = 3;
    switch (variant) {
    case 0: quant_int4_naive<<<ceil_div_l(groups, T), T, 0, st>>>((const __half*)x, scales, mins, packed, groups, g); break;
    case 1: quant_int4_warp<<<ceil_div_l(groups * 32, T), T, 0, st>>>((const __half*)x, scales, mins, packed, groups, g); break;
    case 2: quant_int4_warp_vec<<<ceil_div_l(groups * 32, T), T, 0, st>>>((const __half2*)x, scales, mins, packed, groups, g / 2); break;
    default: quant_int4_warp_reg<<<ceil_div_l(groups * 32, T), T, 0, st>>>((const __half2*)x, scales, mins, packed, groups, g / 2); break;
    }
}

void launch_dequant(const void* payload, void* out, long n, int g, int method, cudaStream_t st) {
    long groups = n / g, n2 = n / 2;
    const int T = 256;
    if (g % 8 == 0 && n % 8 == 0 && (uintptr_t)payload % 16 == 0 && (uintptr_t)out % 16 == 0 &&
        (method != 1 || (2 * groups) % 8 == 0)) {
        dequant_v8<<<grid_for(n / 8, T), T, 0, st>>>((const uint8_t*)payload, (uint4*)out, groups, n / 8, g, method);
        return;
    }
    if (method == 1) dequant_int8<<<grid_for(n2, T), T, 0, st>>>((const uint8_t*)payload, (__half2*)out, groups, n2, g);
    else dequant_int4<<<grid_for(n2, T), T, 0, st>>>((const uint8_t*)payload, (__half2*)out, groups, n2, g);
}

void launch_dequant_scatter(const void* base, const long* offs, const int* toffs, const int* methods,
                            int nblocks, void* dest, long n, int Tb, int D, int Ttot, int g, cudaStream_t st) {
    long n2 = n / 2, groups = n / g;
    const int T = 256;
    // the caller (BlockLoader) aligns every payload to 16 bytes
    if (D % 8 == 0 && g % 8 == 0 && (2 * groups) % 8 == 0 && (uintptr_t)base % 16 == 0 && (uintptr_t)dest % 16 == 0) {
        dim3 grid(grid_for(n / 8, T, 4096), nblocks);
        dequant_scatter_v8<<<grid, T, 0, st>>>((const uint8_t*)base, offs, toffs, methods, (uint4*)dest,
                                               n / 8, groups, Tb, D, Ttot, g);
        return;
    }
    int gx = grid_for(n2, T, 4096);
    dim3 grid(gx, nblocks);
    dequant_scatter<<<grid, T, 0, st>>>((const uint8_t*)base, offs, toffs, methods, (__half2*)dest,
                                        n2, groups, Tb, D, Ttot, g);
}

#include "kvprof.h"

// every kernel above with the block size its launcher uses (for the resource/occupancy report)
int kvquant_kernel_refs(KernelRef* out, int max) {
    const KernelRef refs[] = {
        {"quant_int8 v0 naive", (const void*)quant_int8_naive, 256},
        {"quant_int8 v1 warp", (const void*)quant_int8_warp, 256},
        {"quant_int8 v2 warp_vec", (const void*)quant_int8_warp_vec, 256},
        {"quant_int8 v3 warp_reg", (const void*)quant_int8_warp_reg, 256},
        {"quant_int8 v4 subwarp<8> (g=64)", (const void*)quant_int8_subwarp<8>, 256},
        {"quant_int8 v4 subwarp<16> (g=128)", (const void*)quant_int8_subwarp<16>, 256},
        {"quant_int4 v0 naive", (const void*)quant_int4_naive, 256},
        {"quant_int4 v1 warp", (const void*)quant_int4_warp, 256},
        {"quant_int4 v2 warp_vec", (const void*)quant_int4_warp_vec, 256},
        {"quant_int4 v3 warp_reg", (const void*)quant_int4_warp_reg, 256},
        {"quant_int4 v4 subwarp<8> (g=64)", (const void*)quant_int4_subwarp<8>, 256},
        {"dequant_int8 (half2)", (const void*)dequant_int8, 256},
        {"dequant_int4 (half2)", (const void*)dequant_int4, 256},
        {"dequant_v8 (8 values/thread)", (const void*)dequant_v8, 256},
        {"dequant_scatter (half2)", (const void*)dequant_scatter, 256},
        {"dequant_scatter_v8", (const void*)dequant_scatter_v8, 256},
    };
    int n = 0;
    for (const auto& r : refs) if (n < max) out[n++] = r;
    return n;
}

void query_memory_bandwidth(int device, double* peak_gbps, int* clock_khz, int* bus_bits) {
    cudaDeviceGetAttribute(clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(bus_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    *peak_gbps = 2.0 * (*clock_khz) * 1e3 * (*bus_bits / 8.0) / 1e9;   // DDR: 2 transfers/clock
}
