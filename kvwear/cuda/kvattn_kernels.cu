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

// ------------------------------------------------------------------ v4: tensor cores + shared-memory tiles (GQA)
// The profile of v3 (results/cuda/roofline.md) shows it is latency-bound, not bandwidth-bound: with
// R = 7 query heads per KV head it needs 168 registers/thread (3 blocks = 12 warps per SM) and does
// R x log2(D/8) warp shuffles per token. But GQA decode is really a small matrix product per KV head:
//     S = Q K^T  ([R x D] x [D x tile])   and   O += P V  ([R x tile] x [tile x D]),
// so v4 runs both products on the tensor cores (WMMA 16x16x16, fp16 in / fp32 accumulate) with the
// R query heads padded to the 16 rows of an MMA tile:
//   1. each CUDA block (one KV head, one split) walks its tokens in tiles of TILE tokens;
//   2. the int8 K/V tile is dequantized to fp16 into shared memory (8-byte global loads,
//      16-byte shared stores, rows padded by 8 halves so WMMA loads do not bank-conflict);
//   3. while the tensor cores work on tile i, the raw int8 bytes of tile i+1 are already being
//      fetched into registers (software pipelining: the loads are issued before the MMAs);
//   4. online softmax between the two MMAs: 8 threads per query row, max/sum with 3 shuffles,
//      P stored as fp16, the fp32 O accumulator (shared memory) rescaled by exp(m_old - m_new).
// Output is the same (m, l, acc) partial per (query head, split) as v1-v3, merged by decode_combine.
#include <mma.h>

template <int D, int TILE, int NW>
__global__ void __launch_bounds__(NW * 32)
paged_int8_partial_v4(const __half* __restrict__ q, const uint8_t* __restrict__ base, const long* __restrict__ offs,
                      long groups, int layer, int Hkv, int R, int Tb, int g, int T, int tps, float sm_scale,
                      float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int NT = NW * 32;
    constexpr int KP = D + 8, SP = TILE + 4, PP = TILE + 8, OP = D + 4;     // padded leading dimensions
    constexpr int CPR = D / 8;                                              // 8-value chunks per row
    constexpr int CH = TILE * CPR / NT;                                     // chunks per thread per tile
    static_assert(TILE * CPR % NT == 0, "tile must split evenly over the block");
    constexpr int TPR = NT / 16;                                            // softmax threads per query row
    constexpr int CPT = TILE / TPR;                                         // softmax columns per thread
    static_assert(TPR <= 32 && TILE % TPR == 0 && D % TPR == 0, "softmax mapping");
    __shared__ __align__(32) __half sQ[16 * KP];
    __shared__ __align__(32) __half sK[TILE * KP];
    __shared__ __align__(32) __half sV[TILE * KP];
    __shared__ __align__(32) float sS[16 * SP];
    __shared__ __align__(32) __half sP[16 * PP];
    __shared__ __align__(32) float sO[16 * OP];

    const int hk = blockIdx.x, split = blockIdx.y, ns = gridDim.y;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int gpr = D / g;
    const int t0 = split * tps, t1 = min(T, t0 + tps);
    const long kbase = ((long)(layer * 2 + 0) * Hkv + hk) * Tb, vbase = ((long)(layer * 2 + 1) * Hkv + hk) * Tb;

    for (int i = tid; i < 16 * D; i += NT) {                                // Q rows (R real, rest zero)
        const int r = i / D, d = i - r * D;
        sQ[r * KP + d] = r < R ? q[(long)(hk * R + r) * D + d] : __float2half(0.f);
    }
    for (int i = tid; i < 16 * OP; i += NT) sO[i] = 0.f;

    // raw int8 bytes + scales of one tile, held in registers between issue and use
    uint2 kr[CH], vr[CH];
    float ksc[CH], vsc[CH];
    auto fetch = [&](int tb) {
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            const int ch = tid + c * NT, tl = ch / CPR, d0 = (ch - tl * CPR) * 8, t = tb + tl;
            if (t < t1) {
                const int b = t / Tb, tt = t - b * Tb;
                const uint8_t* p = base + offs[b];
                const __half* sc = reinterpret_cast<const __half*>(p);
                const int8_t* qq = reinterpret_cast<const int8_t*>(p + 2 * groups);
                const long rk = kbase + tt, rv = vbase + tt;
                kr[c] = *reinterpret_cast<const uint2*>(qq + rk * D + d0);
                vr[c] = *reinterpret_cast<const uint2*>(qq + rv * D + d0);
                ksc[c] = __half2float(sc[rk * gpr + d0 / g]);
                vsc[c] = __half2float(sc[rv * gpr + d0 / g]);
            } else {
                kr[c] = vr[c] = make_uint2(0, 0);
                ksc[c] = vsc[c] = 0.f;
            }
        }
    };
    auto dequant_store = [&](const uint2 raw, float s, __half* dst) {      // 8 int8 -> 8 fp16, one 16-byte store
        const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
        uint4 o;
        __half2* h = reinterpret_cast<__half2*>(&o);
#pragma unroll
        for (int k = 0; k < 4; ++k) h[k] = __floats2half2_rn((float)b[2 * k] * s, (float)b[2 * k + 1] * s);
        *reinterpret_cast<uint4*>(dst) = o;
    };

    // softmax ownership: TPR consecutive threads per query row (16 rows x TPR = NT threads)
    const int srow = tid / TPR, sj = tid % TPR;
    float m_run = -INFINITY, l_run = 0.f;

    fetch(t0);
    for (int tb = t0; tb < t1; tb += TILE) {
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            const int ch = tid + c * NT, tl = ch / CPR, d0 = (ch - tl * CPR) * 8;
            dequant_store(kr[c], ksc[c], sK + tl * KP + d0);
            dequant_store(vr[c], vsc[c], sV + tl * KP + d0);
        }
        __syncthreads();                                         // (A) tile i in shared memory
        if (tb + TILE < t1) fetch(tb + TILE);                    // tile i+1 loads in flight during the MMAs

        // S = Q K^T : one 16x16 output fragment per (warp, 16 tokens)
        for (int f = warp; f < TILE / 16; f += NW) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int k = 0; k < D / 16; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;   // K^T
                wmma::load_matrix_sync(a, sQ + k * 16, KP);
                wmma::load_matrix_sync(b, sK + f * 16 * KP + k * 16, KP);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sS + f * 16, acc, SP, wmma::mem_row_major);
        }
        __syncthreads();                                         // (B) scores ready

        // online softmax for row srow over this tile
        {
            float s[CPT], mt = -INFINITY;
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                const int col = sj * CPT + c;
                s[c] = (tb + col < t1) ? sS[srow * SP + col] * sm_scale : -INFINITY;
                mt = fmaxf(mt, s[c]);
            }
#pragma unroll
            for (int o = TPR / 2; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(FULL, mt, o));
            const float mnew = fmaxf(m_run, mt), corr = __expf(m_run - mnew);
            float ps = 0.f;
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                const float p = __expf(s[c] - mnew);
                ps += p;
                sP[srow * PP + sj * CPT + c] = __float2half_rn(p);
            }
#pragma unroll
            for (int o = TPR / 2; o > 0; o >>= 1) ps += __shfl_xor_sync(FULL, ps, o);
            l_run = l_run * corr + ps;
            m_run = mnew;
#pragma unroll
            for (int d = sj * (D / TPR); d < (sj + 1) * (D / TPR); ++d) sO[srow * OP + d] *= corr;
        }
        __syncthreads();                                         // (C) P ready, O rescaled

        // O += P V : one 16x16 fragment of O per (warp, 16 columns of D)
        for (int f = warp; f < D / 16; f += NW) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, sO + f * 16, OP, wmma::mem_row_major);
#pragma unroll
            for (int k = 0; k < TILE / 16; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, sP + k * 16, PP);
                wmma::load_matrix_sync(b, sV + k * 16 * KP + f * 16, KP);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sO + f * 16, acc, OP, wmma::mem_row_major);
        }
        __syncthreads();                                         // (D) shared tiles free for tile i+1
    }
    if (srow < R && sj == 0) {
        const long o = (long)(hk * R + srow) * ns + split;
        pm[o] = m_run;
        pl[o] = l_run;
    }
    for (int i = tid; i < R * D; i += NT) {
        const int r = i / D, d = i - r * D;
        pacc[((long)(hk * R + r) * ns + split) * D + d] = sO[r * OP + d];
    }
#endif
}

// ------------------------------------------------------------------ v5: register-resident tensor-core GQA decode
// v4 keeps S, P and the O accumulator in shared memory (WMMA fragments have an opaque layout), so
// every tile pays smem round trips and 4 __syncthreads; profiling shows it latency-bound at ~1/3
// of DRAM bandwidth. v5 is the FlashAttention-2 design written with raw mma.sync PTX, whose fragment
// layouts are documented, so everything stays in registers:
//   * S = Q K^T with mma.m16n8k16: the C fragment of S (rows gid, gid+8; cols 2*tig, 2*tig+1) is
//     exactly the A fragment P needs for O += P V, so P never leaves registers;
//   * online softmax on the fragment: row max/sum over the 4 lanes of a quad (2 shuffles);
//     the row sum l stays thread-local until the end;
//   * O accumulators (16 x D, fp32) live in registers, rescaled in place;
//   * no shared memory and no __syncthreads in the main loop: each warp streams its own
//     16-token chunks, so loads of many warps overlap freely;
//   * int8 K/V go into the MMAs as exact fp16 integers (magic-number conversion with __byte_perm:
//     fp16 bits 0x64xx = 1024 + xx); the per-token scales are applied to S and to P instead of to
//     every K/V element;
//   * head_dim is permuted consistently for Q and K (a dot product does not depend on the order)
//     so each lane fetches its K slice with one 16-byte load and its V slice with 8-byte loads.
// Requires sm_80 (mma.m16n8k16 f16), R <= 16, Tb % 16 == 0, tps % 16 == 0 and a scale group of
// 64 or 128 values (g == D, or D == 128 with g == 64).
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// two bytes of x (sign bit already flipped: u = int8 + 128), selected by `sel`, -> half2 of the int8 values
__device__ __forceinline__ uint32_t u8x2_to_h2(uint32_t x, uint32_t sel) {
    uint32_t h = __byte_perm(x, 0x64646464u, sel);            // fp16 bits 0x64uu = 1024 + u, exactly
    const __half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));   // 1152
    __half2 r = __hsub2(*reinterpret_cast<__half2*>(&h), bias);
    return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

template <int D, int GPR, int NW>
__global__ void __launch_bounds__(NW * 32)
paged_int8_partial_v5(const __half* __restrict__ q, const uint8_t* __restrict__ base, const long* __restrict__ offs,
                      long groups, int layer, int Hkv, int R, int Tb, int T, int tps, float sm_scale,
                      float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr int KS = D / 16;            // k-steps of S = Q K^T
    constexpr int NTO = D / 8;            // n-tiles of O
    constexpr int DG = D / GPR;           // head_dim values per scale group (64 or 128)
    constexpr int KPG = DG / 16;          // k-steps per scale group
    constexpr int NPG = DG / 8;           // O n-tiles per scale group
    constexpr int NU = D / 64;            // 16-byte K loads / 8-byte V loads per token per lane
    constexpr float PSCALE = 256.f;       // keeps p * v_scale out of the fp16 subnormal range
    static_assert(DG == 64 || DG == 128, "scale group must be 64 or 128 values");

    const int hk = blockIdx.x, split = blockIdx.y, ns = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int t0 = split * tps, t1 = min(T, t0 + tps);
    const long kbase = ((long)(layer * 2 + 0) * Hkv + hk) * Tb, vbase = ((long)(layer * 2 + 1) * Hkv + hk) * Tb;

    // Q as A fragments (rows gid, gid+8 = query heads), head_dim permuted: k-step kk, element e in 0..3
    // is physical d = grp*DG + tig*(DG/4) + kl*4 + e  (grp = kk / KPG, kl = kk % KPG)
    uint32_t qa[KS][4];
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
        const int d = (kk / KPG) * DG + tig * (DG / 4) + (kk % KPG) * 4;
        const uint2 x0 = gid < R ? *reinterpret_cast<const uint2*>(q + (long)(hk * R + gid) * D + d) : make_uint2(0, 0);
        const uint2 x1 = gid + 8 < R ? *reinterpret_cast<const uint2*>(q + (long)(hk * R + gid + 8) * D + d)
                                     : make_uint2(0, 0);
        qa[kk][0] = x0.x; qa[kk][1] = x1.x; qa[kk][2] = x0.y; qa[kk][3] = x1.y;
    }
    float o[NTO][4];
#pragma unroll
    for (int nt = 0; nt < NTO; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    for (int tc = t0 + warp * 16; tc < t1; tc += NW * 16) {
        const int b = tc / Tb, tt = tc - b * Tb;                 // the 16 tokens share one paged block
        const uint8_t* p = base + offs[b];
        const __half* sc = reinterpret_cast<const __half*>(p);
        const uint8_t* qq = p + 2 * groups;
        const long rk = kbase + tt, rv = vbase + tt;
        // K: tokens gid (n-tile 0) and 8+gid (n-tile 1); word kk holds the 4 values of k-step kk
        uint32_t kw[2][KS];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                const int kk0 = u * 4, off = (kk0 / KPG) * DG + tig * (DG / 4) + (kk0 % KPG) * 4;
                const uint4 w = *reinterpret_cast<const uint4*>(qq + (rk + j * 8 + gid) * D + off);
                kw[j][kk0] = w.x ^ 0x80808080u; kw[j][kk0 + 1] = w.y ^ 0x80808080u;
                kw[j][kk0 + 2] = w.z ^ 0x80808080u; kw[j][kk0 + 3] = w.w ^ 0x80808080u;
            }
        // V: tokens 2*tig, 2*tig+1, 8+2*tig, 9+2*tig (the k index of the P V product); byte nt of a
        // token's words is physical d = grp*DG + gid*(DG/8) + nt % NPG  (grp = nt / NPG)
        uint32_t vw[4][2 * NU];
        const int tv[4] = {2 * tig, 2 * tig + 1, 8 + 2 * tig, 9 + 2 * tig};
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                const int nt0 = u * 8, off = (nt0 / NPG) * DG + gid * (DG / 8) + nt0 % NPG;
                const uint2 w = *reinterpret_cast<const uint2*>(qq + (rv + tv[i]) * D + off);
                vw[i][2 * u] = w.x ^ 0x80808080u; vw[i][2 * u + 1] = w.y ^ 0x80808080u;
            }
        // scales of this lane's 4 tokens (S columns and P V k-index are the same tokens)
        float ks[4][GPR], vs[4][GPR];
        bool ok[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ok[i] = tc + tv[i] < t1;
#pragma unroll
            for (int gq = 0; gq < GPR; ++gq) {
                ks[i][gq] = ok[i] ? __half2float(sc[(rk + tv[i]) * GPR + gq]) * sm_scale : 0.f;
                vs[i][gq] = ok[i] ? __half2float(sc[(rv + tv[i]) * GPR + gq]) * PSCALE : 0.f;
            }
        }

        // S = Q K^T per scale group, two n-tiles of 8 tokens
        float s[2][GPR][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int gq = 0; gq < GPR; ++gq) s[j][gq][0] = s[j][gq][1] = s[j][gq][2] = s[j][gq][3] = 0.f;
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int kk = 0; kk < KS; ++kk)
                mma16816(s[j][kk / KPG], qa[kk], u8x2_to_h2(kw[j][kk], 0x4140), u8x2_to_h2(kw[j][kk], 0x4342));

        // scores of rows gid (x0) and gid+8 (x1) for tokens tv[0..3]
        float x0[4], x1[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = i >> 1, e = i & 1;
            float a = 0.f, c = 0.f;
#pragma unroll
            for (int gq = 0; gq < GPR; ++gq) { a += s[j][gq][e] * ks[i][gq]; c += s[j][gq][2 + e] * ks[i][gq]; }
            x0[i] = ok[i] ? a : -INFINITY;
            x1[i] = ok[i] ? c : -INFINITY;
        }
        float mt0 = fmaxf(fmaxf(x0[0], x0[1]), fmaxf(x0[2], x0[3]));
        float mt1 = fmaxf(fmaxf(x1[0], x1[1]), fmaxf(x1[2], x1[3]));
        mt0 = fmaxf(mt0, __shfl_xor_sync(FULL, mt0, 1)); mt0 = fmaxf(mt0, __shfl_xor_sync(FULL, mt0, 2));
        mt1 = fmaxf(mt1, __shfl_xor_sync(FULL, mt1, 1)); mt1 = fmaxf(mt1, __shfl_xor_sync(FULL, mt1, 2));
        const float mn0 = fmaxf(m0, mt0), mn1 = fmaxf(m1, mt1);
        const float c0 = __expf(m0 - mn0), c1 = __expf(m1 - mn1);
        float p0[4], p1[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { p0[i] = __expf(x0[i] - mn0); p1[i] = __expf(x1[i] - mn1); }
        l0 = l0 * c0 + (p0[0] + p0[1] + p0[2] + p0[3]);
        l1 = l1 * c1 + (p1[0] + p1[1] + p1[2] + p1[3]);
        m0 = mn0; m1 = mn1;
#pragma unroll
        for (int nt = 0; nt < NTO; ++nt) { o[nt][0] *= c0; o[nt][1] *= c0; o[nt][2] *= c1; o[nt][3] *= c1; }

        // O += P' V with P' = p * v_scale (per token, per scale group) as the A fragment
#pragma unroll
        for (int gq = 0; gq < GPR; ++gq) {
            const uint32_t pa[4] = {pack_h2(p0[0] * vs[0][gq], p0[1] * vs[1][gq]),
                                    pack_h2(p1[0] * vs[0][gq], p1[1] * vs[1][gq]),
                                    pack_h2(p0[2] * vs[2][gq], p0[3] * vs[3][gq]),
                                    pack_h2(p1[2] * vs[2][gq], p1[3] * vs[3][gq])};
#pragma unroll
            for (int nl = 0; nl < NPG; ++nl) {
                const int nt = gq * NPG + nl, wi = nt >> 2, by = nt & 3;
                const uint32_t sel = 0x0040u | by | ((4u + by) << 4);              // [x.by, y.by]
                const uint32_t b0 = u8x2_to_h2(__byte_perm(vw[0][wi], vw[1][wi], sel), 0x4140);
                const uint32_t b1 = u8x2_to_h2(__byte_perm(vw[2][wi], vw[3][wi], sel), 0x4140);
                mma16816(o[nt], pa, b0, b1);
            }
        }
    }
    // finish the row sums across the quad, then merge the NW warps' states. The merge buffer is a
    // single 16 x D tile that the warps accumulate into in turn (NW-1 extra barriers, epilogue only):
    // 4x less shared memory than one tile per warp, which is what limits occupancy at D = 128.
    l0 += __shfl_xor_sync(FULL, l0, 1); l0 += __shfl_xor_sync(FULL, l0, 2);
    l1 += __shfl_xor_sync(FULL, l1, 1); l1 += __shfl_xor_sync(FULL, l1, 2);
    __shared__ float smax[NW][16], ssum[NW][16];
    __shared__ float so[16][D];
    if (tig == 0) { smax[warp][gid] = m0; smax[warp][gid + 8] = m1; ssum[warp][gid] = l0; ssum[warp][gid + 8] = l1; }
    __syncthreads();
    float M0 = -INFINITY, M1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < NW; ++w) { M0 = fmaxf(M0, smax[w][gid]); M1 = fmaxf(M1, smax[w][gid + 8]); }
    const float f0 = m0 == -INFINITY ? 0.f : __expf(m0 - M0), f1 = m1 == -INFINITY ? 0.f : __expf(m1 - M1);
    for (int w = 0; w < NW; ++w) {
        if (warp == w) {
#pragma unroll
            for (int nt = 0; nt < NTO; ++nt) {
                const int db = (nt / NPG) * DG + nt % NPG;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int d = db + (2 * tig + e) * (DG / 8);
                    so[gid][d] = (w ? so[gid][d] : 0.f) + o[nt][e] * f0;
                    so[gid + 8][d] = (w ? so[gid + 8][d] : 0.f) + o[nt][2 + e] * f1;
                }
            }
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < R * D; i += NW * 32) {
        const int r = i / D, d = i - r * D;
        const long oi = (long)(hk * R + r) * ns + split;
        if (d == 0) {
            float M = -INFINITY, L = 0.f;
#pragma unroll
            for (int w = 0; w < NW; ++w) M = fmaxf(M, smax[w][r]);
#pragma unroll
            for (int w = 0; w < NW; ++w) L += smax[w][r] == -INFINITY ? 0.f : ssum[w][r] * __expf(smax[w][r] - M);
            pm[oi] = M;
            pl[oi] = L;
        }
        pacc[oi * D + d] = so[r][d] * (1.f / PSCALE);
    }
#endif
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
    const int R = Hq / Hkv;
    const bool gqa16 = R > 1 && R <= 16 && g % 8 == 0 && (2 * groups) % 8 == 0;
    // v5 (mma.sync, register-resident): GQA with R <= 16, 64/128-value scale groups, 16-token alignment
    if (variant >= 5 && gqa16 && (D == 64 || D == 128) && (g == D || (D == 128 && g == 64)) && Tb % 16 == 0 &&
        tps % 16 == 0) {
        int ns = (T + tps - 1) / tps;
        dim3 grid(Hkv, ns);
        const __half* qh = reinterpret_cast<const __half*>(q);
        const uint8_t* bp = (const uint8_t*)base;
        if (D == 64)
            paged_int8_partial_v5<64, 1, 4><<<grid, 128, 0, st>>>(qh, bp, offs, groups, layer, Hkv, R, Tb, T, tps, scale,
                                                                pm, pl, pacc);
        else if (g == 128)
            paged_int8_partial_v5<128, 1, 4><<<grid, 128, 0, st>>>(qh, bp, offs, groups, layer, Hkv, R, Tb, T, tps, scale,
                                                                 pm, pl, pacc);
        else
            paged_int8_partial_v5<128, 2, 4><<<grid, 128, 0, st>>>(qh, bp, offs, groups, layer, Hkv, R, Tb, T, tps, scale,
                                                                 pm, pl, pacc);
        decode_combine<<<Hq, D, 0, st>>>(pm, pl, pacc, (__half*)out, ns, D);
        return;
    }
    // v4 (WMMA tensor cores, shared-memory tiles): GQA with R <= 16 query heads per KV head
    if (variant >= 4 && gqa16) {
        int ns = (T + tps - 1) / tps;
        dim3 grid(Hkv, ns);
        const __half* qh = reinterpret_cast<const __half*>(q);
        const uint8_t* bp = (const uint8_t*)base;
        if (D == 64)
            paged_int8_partial_v4<64, 64, 8><<<grid, 256, 0, st>>>(qh, bp, offs, groups, layer, Hkv, R, Tb, g, T, tps,
                                                                  scale, pm, pl, pacc);
        else
            paged_int8_partial_v4<128, 32, 8><<<grid, 256, 0, st>>>(qh, bp, offs, groups, layer, Hkv, R, Tb, g, T, tps,
                                                                   scale, pm, pl, pacc);
        decode_combine<<<Hq, D, 0, st>>>(pm, pl, pacc, (__half*)out, ns, D);
        return;
    }
    // v3 only pays off with GQA (R > 1); for R == 1 it is v2
    if (variant >= 3 && g % 8 == 0 && (2 * groups) % 8 == 0 && Hq / Hkv > 1 && Hq / Hkv <= MAX_R) {
        int ns = (T + tps - 1) / tps;
        dim3 grid(Hkv, ns);
        const __half* qh = reinterpret_cast<const __half*>(q);
        const uint8_t* bp = (const uint8_t*)base;
#define V3(DD, RR)                                                                                \
    paged_int8_partial_v3<DD, RR><<<grid, NWARPS * 32, 0, st>>>(qh, bp, offs, groups, layer, Hq, Hkv, \
                                                                Tb, g, T, tps, scale, pm, pl, pacc)
#define V3R(DD)                                                                                   \
    switch (Hq / Hkv) {                                                                           \
    case 2: V3(DD, 2); break;                                                                     \
    case 3: V3(DD, 3); break;                                                                     \
    case 4: V3(DD, 4); break;                                                                     \
    case 5: V3(DD, 5); break;                                                                     \
    case 6: V3(DD, 6); break;                                                                     \
    case 7: V3(DD, 7); break;                                                                     \
    default: V3(DD, 8); break;                                                                    \
    }
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

#include "kvprof.h"

int kvattn_kernel_refs(KernelRef* out, int max) {
    const int B = NWARPS * 32;
    const KernelRef refs[] = {
        {"attn paged_int8 v1 <D=64>", (const void*)decode_partial<2, PagedInt8>, B},
        {"attn paged_int8 v1 <D=128>", (const void*)decode_partial<4, PagedInt8>, B},
        {"attn dense_fp16 <D=64>", (const void*)decode_partial<2, DenseFp16>, B},
        {"attn dense_fp16 <D=128>", (const void*)decode_partial<4, DenseFp16>, B},
        {"attn paged_int8 v2 <D=64>", (const void*)paged_int8_partial_v2<64>, B},
        {"attn paged_int8 v2 <D=128>", (const void*)paged_int8_partial_v2<128>, B},
        {"attn paged_int8 v3 <D=64,R=2>", (const void*)paged_int8_partial_v3<64, 2>, B},
        {"attn paged_int8 v3 <D=64,R=4>", (const void*)paged_int8_partial_v3<64, 4>, B},
        {"attn paged_int8 v3 <D=64,R=7> (Qwen2.5-0.5B)", (const void*)paged_int8_partial_v3<64, 7>, B},
        {"attn paged_int8 v3 <D=64,R=8>", (const void*)paged_int8_partial_v3<64, 8>, B},
        {"attn paged_int8 v3 <D=128,R=4>", (const void*)paged_int8_partial_v3<128, 4>, B},
        {"attn paged_int8 v3 <D=128,R=8>", (const void*)paged_int8_partial_v3<128, 8>, B},
        {"attn paged_int8 v4 WMMA <D=64,TILE=64,NW=8>", (const void*)paged_int8_partial_v4<64, 64, 8>, 256},
        {"attn paged_int8 v4 WMMA <D=128,TILE=32,NW=8>", (const void*)paged_int8_partial_v4<128, 32, 8>, 256},
        {"attn paged_int8 v5 mma.sync <D=64,g=64>", (const void*)paged_int8_partial_v5<64, 1, 4>, B},
        {"attn paged_int8 v5 mma.sync <D=128,g=128>", (const void*)paged_int8_partial_v5<128, 1, 4>, B},
        {"attn paged_int8 v5 mma.sync <D=128,g=64>", (const void*)paged_int8_partial_v5<128, 2, 4>, B},
        {"attn decode_combine <D=64>", (const void*)decode_combine, 64},
    };
    int n = 0;
    for (const auto& r : refs) if (n < max) out[n++] = r;
    return n;
}
