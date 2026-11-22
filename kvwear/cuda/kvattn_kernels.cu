// Decode-step attention that reads the KV cache *directly in kvwear's int8 block format*,
// through a block table (paged, as in vLLM's PagedAttention), without dequantizing first.
//
// Why: kvwear's int8 block payload stores one fp16 scale per (layer, K/V, head, token) row of
// head_dim values. That is exactly the granularity attention needs, so a block read from the SSD
// can be attended to as-is: half the bytes per decode step (decode attention is memory-bound)
// and twice the context in the same GPU memory.
//
// Algorithm: flash-decoding (split-K).
//   kernel 1  grid (q_heads, splits). Each CUDA block handles one head and one contiguous chunk
//             of tokens. Each warp walks tokens of the chunk; per token the 32 lanes compute the
//             q.k dot product (D/32 values per lane, warp-shuffle sum), then update an online
//             softmax (running max m, normaliser l, weighted V accumulator) in registers.
//             Warps are merged through shared memory -> one partial (m, l, acc[D]) per split.
//   kernel 2  merges the splits of each head with the same log-sum-exp rescaling.
// GQA: q head h reads kv head h / (Hq / Hkv).
// A dense fp16 [Hkv, T, D] variant of kernel 1 is the like-for-like baseline.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FULL 0xffffffffu
#define NWARPS 4

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

struct PagedInt8 {
    const uint8_t* base; const long* offs; long groups;   // scale groups per block = layers*2*Hkv*Tb*gpr
    int layer, Hkv, Tb, g;                                 // g: values per scale group (divides D)
    template <int E>
    __device__ __forceinline__ void load(int hk, int t, int lane, float (&k)[E], float (&v)[E]) const {
        int b = t / Tb, tt = t - b * Tb;
        const uint8_t* p = base + offs[b];
        const __half* sc = reinterpret_cast<const __half*>(p);
        const int8_t* q = reinterpret_cast<const int8_t*>(p + 2 * groups);
        long rk = ((long)(layer * 2 + 0) * Hkv + hk) * Tb + tt;   // row = (layer, K, head, token)
        long rv = ((long)(layer * 2 + 1) * Hkv + hk) * Tb + tt;
        const int D = 32 * E;
        const int gpr = D / g, sub = (lane * E) / g;             // this lane's values share one group
        float ks = __half2float(sc[rk * gpr + sub]), vs = __half2float(sc[rv * gpr + sub]);
        if constexpr (E == 2) {
            char2 a = reinterpret_cast<const char2*>(q + rk * D)[lane];
            char2 c = reinterpret_cast<const char2*>(q + rv * D)[lane];
            k[0] = a.x * ks; k[1] = a.y * ks; v[0] = c.x * vs; v[1] = c.y * vs;
        } else {
            char4 a = reinterpret_cast<const char4*>(q + rk * D)[lane];
            char4 c = reinterpret_cast<const char4*>(q + rv * D)[lane];
            k[0] = a.x * ks; k[1] = a.y * ks; k[2] = a.z * ks; k[3] = a.w * ks;
            v[0] = c.x * vs; v[1] = c.y * vs; v[2] = c.z * vs; v[3] = c.w * vs;
        }
    }
};

struct DenseFp16 {
    const __half* K; const __half* V; int T;               // [Hkv, T, D]
    template <int E>
    __device__ __forceinline__ void load(int hk, int t, int lane, float (&k)[E], float (&v)[E]) const {
        const int D = 32 * E;
        long row = ((long)hk * T + t) * D;
        if constexpr (E == 2) {
            float2 a = __half22float2(reinterpret_cast<const __half2*>(K + row)[lane]);
            float2 c = __half22float2(reinterpret_cast<const __half2*>(V + row)[lane]);
            k[0] = a.x; k[1] = a.y; v[0] = c.x; v[1] = c.y;
        } else {
            const __half2* kp = reinterpret_cast<const __half2*>(K + row) + 2 * lane;
            const __half2* vp = reinterpret_cast<const __half2*>(V + row) + 2 * lane;
            float2 a0 = __half22float2(kp[0]), a1 = __half22float2(kp[1]);
            float2 c0 = __half22float2(vp[0]), c1 = __half22float2(vp[1]);
            k[0] = a0.x; k[1] = a0.y; k[2] = a1.x; k[3] = a1.y;
            v[0] = c0.x; v[1] = c0.y; v[2] = c1.x; v[3] = c1.y;
        }
    }
};

template <int E, class KV>
__global__ void __launch_bounds__(NWARPS * 32)
decode_partial(const __half* __restrict__ q, KV kv, int Hq, int Hkv, int T, int tps, float sm_scale,
               float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    constexpr int D = 32 * E;
    const int h = blockIdx.x, split = blockIdx.y, ns = gridDim.y;
    const int hk = h / (Hq / Hkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float qv[E];
#pragma unroll
    for (int e = 0; e < E; ++e) qv[e] = __half2float(q[(long)h * D + lane * E + e]) * sm_scale;

    const int t0 = split * tps, t1 = min(T, t0 + tps);
    float m = -INFINITY, l = 0.f, acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.f;
    for (int t = t0 + warp; t < t1; t += NWARPS) {
        float k[E], v[E];
        kv.template load<E>(hk, t, lane, k, v);
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) d += qv[e] * k[e];
        d = wsum(d);                                   // every lane now holds q.k
        float mn = fmaxf(m, d);
        float corr = __expf(m - mn), p = __expf(d - mn);
        l = l * corr + p;
#pragma unroll
        for (int e = 0; e < E; ++e) acc[e] = acc[e] * corr + p * v[e];
        m = mn;
    }
    // merge the NWARPS partial softmaxes of this CUDA block
    __shared__ float sm[NWARPS], sl[NWARPS], sacc[NWARPS][D];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for (int e = 0; e < E; ++e) sacc[warp][lane * E + e] = acc[e];
    __syncthreads();
    if (warp == 0) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f, f[NWARPS];
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) { f[w] = (sm[w] == -INFINITY) ? 0.f : __expf(sm[w] - M); L += sl[w] * f[w]; }
        long o = (long)h * ns + split;
        if (lane == 0) { pm[o] = M; pl[o] = L; }
#pragma unroll
        for (int e = 0; e < E; ++e) {
            int d = lane * E + e;
            float a = 0.f;
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) a += sacc[w][d] * f[w];
            pacc[o * D + d] = a;
        }
    }
}


// ------------------------------------------------------------------ v2: sub-warp tokens, 8-byte int8 loads
// v1 (above) gives each warp one token at a time: 2-byte loads per lane and a 5-step shuffle
// chain per token, so it is latency-bound (~half of DRAM bandwidth). v2 gives each token a
// sub-warp of LPT = D/8 lanes (8 int8 values = one 8-byte load per lane), so a warp has
// TPW = 32/LPT tokens in flight and reduces over log2(LPT) steps. Each sub-warp keeps its own
// online-softmax state; the NWARPS*TPW states are merged in shared memory at the end.
// The loop trip count is uniform across the warp, so full-mask shuffles stay legal.
template <int D>
__global__ void __launch_bounds__(NWARPS * 32)
paged_int8_partial_v2(const __half* __restrict__ q, const uint8_t* __restrict__ base, const long* __restrict__ offs,
                      long groups, int layer, int Hq, int Hkv, int Tb, int g, int T, int tps, float sm_scale,
                      float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    constexpr int LPT = D / 8, TPW = 32 / LPT, NS = NWARPS * TPW;
    const int h = blockIdx.x, split = blockIdx.y, ns = gridDim.y;
    const int hk = h / (Hq / Hkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int sub = lane / LPT, sl = lane % LPT;          // token slot within the warp, lane within it
    const int d0 = sl * 8;
    const int gpr = D / g, gsub = d0 / g;
    float qv[8];
    {
        uint4 r = *reinterpret_cast<const uint4*>(q + (long)h * D + d0);
        const __half2* hh = reinterpret_cast<const __half2*>(&r);
#pragma unroll
        for (int k = 0; k < 4; ++k) { float2 t = __half22float2(hh[k]); qv[2 * k] = t.x * sm_scale; qv[2 * k + 1] = t.y * sm_scale; }
    }
    const int t0 = split * tps, t1 = min(T, t0 + tps);
    float m = -INFINITY, l = 0.f, acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
    const long kbase = ((long)(layer * 2 + 0) * Hkv + hk) * Tb, vbase = ((long)(layer * 2 + 1) * Hkv + hk) * Tb;
    for (int tb = t0 + warp * TPW; tb < t1; tb += NWARPS * TPW) {
        const int t = tb + sub;
        const bool ok = t < t1;
        float d = 0.f, kvs = 0.f;
        uint2 vraw = make_uint2(0, 0);
        if (ok) {
            const int b = t / Tb, tt = t - b * Tb;
            const uint8_t* p = base + offs[b];
            const __half* sc = reinterpret_cast<const __half*>(p);
            const int8_t* qq = reinterpret_cast<const int8_t*>(p + 2 * groups);
            const long rk = kbase + tt, rv = vbase + tt;
            uint2 kraw = *reinterpret_cast<const uint2*>(qq + rk * D + d0);
            vraw = *reinterpret_cast<const uint2*>(qq + rv * D + d0);
            const float ks = __half2float(sc[rk * gpr + gsub]);
            kvs = __half2float(sc[rv * gpr + gsub]);
            const int8_t* kb = reinterpret_cast<const int8_t*>(&kraw);
#pragma unroll
            for (int e = 0; e < 8; ++e) d += qv[e] * (float)kb[e];
            d *= ks;
        }
#pragma unroll
        for (int o = LPT / 2; o > 0; o >>= 1) d += __shfl_xor_sync(FULL, d, o);
        if (ok) {
            const float mn = fmaxf(m, d), corr = __expf(m - mn), pexp = __expf(d - mn);
            l = l * corr + pexp;
            const int8_t* vb = reinterpret_cast<const int8_t*>(&vraw);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = acc[e] * corr + pexp * ((float)vb[e] * kvs);
            m = mn;
        }
    }
    __shared__ float sm[NS], sl_[NS], sacc[NS][D];
    const int slot = warp * TPW + sub;
    if (sl == 0) { sm[slot] = m; sl_[slot] = l; }
#pragma unroll
    for (int e = 0; e < 8; ++e) sacc[slot][d0 + e] = acc[e];
    __syncthreads();
    if (warp == 0) {
        float M = -INFINITY;
        for (int w = 0; w < NS; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f;
        for (int w = 0; w < NS; ++w) L += (sm[w] == -INFINITY) ? 0.f : sl_[w] * __expf(sm[w] - M);
        const long o = (long)h * ns + split;
        if (lane == 0) { pm[o] = M; pl[o] = L; }
        for (int dd = lane; dd < D; dd += 32) {
            float a = 0.f;
            for (int w = 0; w < NS; ++w) a += (sm[w] == -INFINITY) ? 0.f : sacc[w][dd] * __expf(sm[w] - M);
            pacc[o * D + dd] = a;
        }
    }
}

// ------------------------------------------------------------------ v3: v2 + GQA head packing
// With grouped-query attention, R = Hq/Hkv query heads read the same KV head. v1/v2 launch one
// CUDA block per *query* head, so each K/V row is fetched R times (from L2 at best). v3 launches
// one CUDA block per *KV* head and split: every K/V row is loaded once and scored against all R
// query heads held in registers, each with its own online-softmax state. R <= MAX_R.
#define MAX_R 8
template <int D, int R>
__global__ void __launch_bounds__(NWARPS * 32)
paged_int8_partial_v3(const __half* __restrict__ q, const uint8_t* __restrict__ base, const long* __restrict__ offs,
                      long groups, int layer, int Hq, int Hkv, int Tb, int g, int T, int tps, float sm_scale,
                      float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    constexpr int LPT = D / 8, TPW = 32 / LPT, NS = NWARPS * TPW;
    const int hk = blockIdx.x, split = blockIdx.y, ns = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int sub = lane / LPT, sl = lane % LPT, d0 = sl * 8;
    const int gpr = D / g, gsub = d0 / g;
    float qv[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        {
            uint4 raw = *reinterpret_cast<const uint4*>(q + (long)(hk * R + r) * D + d0);
            const __half2* hh = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
            for (int k = 0; k < 4; ++k) { float2 t = __half22float2(hh[k]); qv[r][2 * k] = t.x * sm_scale; qv[r][2 * k + 1] = t.y * sm_scale; }
        }
    }
    const int t0 = split * tps, t1 = min(T, t0 + tps);
    float m[R], l[R], acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        m[r] = -INFINITY; l[r] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[r][e] = 0.f;
    }
    const long kbase = ((long)(layer * 2 + 0) * Hkv + hk) * Tb, vbase = ((long)(layer * 2 + 1) * Hkv + hk) * Tb;
    for (int tb = t0 + warp * TPW; tb < t1; tb += NWARPS * TPW) {
        const int t = tb + sub;
        const bool ok = t < t1;
        float kf[8], vf[8], ks = 0.f, vs = 0.f;
        if (ok) {
            const int b = t / Tb, tt = t - b * Tb;
            const uint8_t* p = base + offs[b];
            const __half* sc = reinterpret_cast<const __half*>(p);
            const int8_t* qq = reinterpret_cast<const int8_t*>(p + 2 * groups);
            const long rk = kbase + tt, rv = vbase + tt;
            uint2 kraw = *reinterpret_cast<const uint2*>(qq + rk * D + d0);
            uint2 vraw = *reinterpret_cast<const uint2*>(qq + rv * D + d0);
            ks = __half2float(sc[rk * gpr + gsub]);
            vs = __half2float(sc[rv * gpr + gsub]);
            const int8_t* kb = reinterpret_cast<const int8_t*>(&kraw);
            const int8_t* vb = reinterpret_cast<const int8_t*>(&vraw);
#pragma unroll
            // scale per lane, BEFORE the reduction: with D/g > 1 groups per row, lanes of one
            // token hold values from different scale groups
            for (int e = 0; e < 8; ++e) { kf[e] = (float)kb[e] * ks; vf[e] = (float)vb[e] * vs; }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) { kf[e] = 0.f; vf[e] = 0.f; }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            {
                float d = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) d += qv[r][e] * kf[e];
#pragma unroll
                for (int o = LPT / 2; o > 0; o >>= 1) d += __shfl_xor_sync(FULL, d, o);
                if (ok) {
                    const float mn = fmaxf(m[r], d), corr = __expf(m[r] - mn), pexp = __expf(d - mn);
                    l[r] = l[r] * corr + pexp;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[r][e] = acc[r][e] * corr + pexp * vf[e];
                    m[r] = mn;
                }
            }
        }
    }
    // merge NS sub-warp states per query head; shared memory reused head by head
    __shared__ float sm[NS], sl_[NS], sacc[NS][D];
    const int slot = warp * TPW + sub;
    for (int r = 0; r < R; ++r) {
        float mr = -INFINITY, lr = 0.f, ar[8];
#pragma unroll
        for (int rr = 0; rr < R; ++rr)
            if (rr == r) {
                mr = m[rr]; lr = l[rr];
#pragma unroll
                for (int e = 0; e < 8; ++e) ar[e] = acc[rr][e];
            }
        if (sl == 0) { sm[slot] = mr; sl_[slot] = lr; }
#pragma unroll
        for (int e = 0; e < 8; ++e) sacc[slot][d0 + e] = ar[e];
        __syncthreads();
        if (warp == 0) {
            float M = -INFINITY;
            for (int w = 0; w < NS; ++w) M = fmaxf(M, sm[w]);
            float L = 0.f;
            for (int w = 0; w < NS; ++w) L += (sm[w] == -INFINITY) ? 0.f : sl_[w] * __expf(sm[w] - M);
            const long o = (long)(hk * R + r) * ns + split;
            if (lane == 0) { pm[o] = M; pl[o] = L; }
            for (int dd = lane; dd < D; dd += 32) {
                float a = 0.f;
                for (int w = 0; w < NS; ++w) a += (sm[w] == -INFINITY) ? 0.f : sacc[w][dd] * __expf(sm[w] - M);
                pacc[o * D + dd] = a;
            }
        }
        __syncthreads();
    }
}

__global__ void decode_combine(const float* __restrict__ pm, const float* __restrict__ pl,
                               const float* __restrict__ pacc, __half* __restrict__ out, int ns, int D) {
    const int h = blockIdx.x, d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, pm[(long)h * ns + s]);
    float L = 0.f, a = 0.f;
    for (int s = 0; s < ns; ++s) {
        long o = (long)h * ns + s;
        float f = (pm[o] == -INFINITY) ? 0.f : __expf(pm[o] - M);
        L += pl[o] * f;
        a += pacc[o * D + d] * f;
    }
    out[(long)h * D + d] = __float2half_rn(a / L);
}

template <class KV>
static void run(const __half* q, KV kv, int Hq, int Hkv, int T, int D, int tps, float scale,
                float* pm, float* pl, float* pacc, __half* out, cudaStream_t st) {
    int ns = (T + tps - 1) / tps;
    dim3 grid(Hq, ns);
    if (D == 64) decode_partial<2, KV><<<grid, NWARPS * 32, 0, st>>>(q, kv, Hq, Hkv, T, tps, scale, pm, pl, pacc);
    else decode_partial<4, KV><<<grid, NWARPS * 32, 0, st>>>(q, kv, Hq, Hkv, T, tps, scale, pm, pl, pacc);
    decode_combine<<<Hq, D, 0, st>>>(pm, pl, pacc, out, ns, D);
}

void launch_paged_int8_decode(const void* q, const void* base, const long* offs, long groups, int layer,
                              int Hq, int Hkv, int Tb, int T, int D, int g, int tps, float scale,
                              float* pm, float* pl, float* pacc, void* out, int variant, cudaStream_t st) {
    // v3 only pays off with GQA (R > 1); for R == 1 it is v2
    if (variant >= 3 && g % 8 == 0 && (2 * groups) % 8 == 0 && Hq / Hkv > 1 && Hq / Hkv <= MAX_R) {
        int ns = (T + tps - 1) / tps;
        dim3 grid(Hkv, ns);
        const __half* qh = reinterpret_cast<const __half*>(q);
        const uint8_t* bp = (const uint8_t*)base;
#define V3(DD, RR) paged_int8_partial_v3<DD, RR><<<grid, NWARPS * 32, 0, st>>>(qh, bp, offs, groups, layer, Hq, Hkv,                                                                         Tb, g, T, tps, scale, pm, pl, pacc)
#define V3R(DD)                                                                                          switch (Hq / Hkv) {                                                                              case 2: V3(DD, 2); break; case 3: V3(DD, 3); break; case 4: V3(DD, 4); break;                    case 5: V3(DD, 5); break; case 6: V3(DD, 6); break; case 7: V3(DD, 7); break;                    default: V3(DD, 8); break;                                                                       }
        if (D == 64) { V3R(64) } else { V3R(128) }
#undef V3R
#undef V3
        decode_combine<<<Hq, D, 0, st>>>(pm, pl, pacc, (__half*)out, ns, D);
        return;
    }
    if (variant >= 2 && g % 8 == 0 && (2 * groups) % 8 == 0) {   // 8-byte loads need 8-byte alignment
        int ns = (T + tps - 1) / tps;
        dim3 grid(Hq, ns);
        const __half* qh = reinterpret_cast<const __half*>(q);
        if (D == 64)
            paged_int8_partial_v2<64><<<grid, NWARPS * 32, 0, st>>>(qh, (const uint8_t*)base, offs, groups, layer, Hq, Hkv,
                                                                   Tb, g, T, tps, scale, pm, pl, pacc);
        else
            paged_int8_partial_v2<128><<<grid, NWARPS * 32, 0, st>>>(qh, (const uint8_t*)base, offs, groups, layer, Hq, Hkv,
                                                                    Tb, g, T, tps, scale, pm, pl, pacc);
        decode_combine<<<Hq, D, 0, st>>>(pm, pl, pacc, (__half*)out, ns, D);
        return;
    }
    PagedInt8 kv{(const uint8_t*)base, offs, groups, layer, Hkv, Tb, g};
    run(reinterpret_cast<const __half*>(q), kv, Hq, Hkv, T, D, tps, scale, pm, pl, pacc, (__half*)out, st);
}

void launch_dense_fp16_decode(const void* q, const void* K, const void* V, int Hq, int Hkv, int T, int D,
                              int tps, float scale, float* pm, float* pl, float* pacc, void* out, cudaStream_t st) {
    DenseFp16 kv{(const __half*)K, (const __half*)V, T};
    run(reinterpret_cast<const __half*>(q), kv, Hq, Hkv, T, D, tps, scale, pm, pl, pacc, (__half*)out, st);
}
