// Kernel registry used by the profiling tools (kvprof_kernels.cu, kvwear.cuda.roofline).
// Each .cu file lists its kernels with the block size its launcher uses, so the static
// resource report (registers, spills, shared memory, occupancy) covers exactly what runs.
#pragma once

struct KernelRef {
    const char* name;      // readable name, e.g. "quant_int8 v4 subwarp<8>"
    const void* fn;        // host stub of the __global__ function
    int block;             // threads per block used by the launcher
};

struct KernelAttr {
    const char* name;
    int block;
    int regs;              // registers per thread (ptxas allocation)
    int local_bytes;       // local memory per thread: register spills + stack arrays
    int static_smem;       // __shared__ bytes per block
    int const_bytes;
    int max_threads;       // max threads per block this kernel can be launched with
    int ptx_version, binary_version;
    int blocks_per_sm;     // cudaOccupancyMaxActiveBlocksPerMultiprocessor at `block`
};

int kvquant_kernel_refs(KernelRef* out, int max);
int kvattn_kernel_refs(KernelRef* out, int max);
int kvprof_kernel_refs(KernelRef* out, int max);
