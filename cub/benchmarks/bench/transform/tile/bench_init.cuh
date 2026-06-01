// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace bench_init {

// splitmix64 — fast deterministic PRNG, one mix per element.
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Map a uint64 to a "reasonable" finite value of T in roughly [-1, 1) for floats,
// or to a non-zero byte for small ints (so neither all-zero nor pathological).
template <typename T>
__device__ __forceinline__ T from_random(uint64_t r) {
    if constexpr (std::is_same_v<T, float>) {
        // 24-bit mantissa precision, range (-1, 1)
        uint32_t u = uint32_t(r >> 40);                // 24 bits
        float f = float(u) * (1.0f / float(1u << 23)) - 1.0f;
        return f;
    } else if constexpr (std::is_same_v<T, double>) {
        uint64_t u = r >> 11;                          // 53 bits
        double d = double(u) * (1.0 / double(1ull << 52)) - 1.0;
        return d;
    } else if constexpr (std::is_same_v<T, __half>) {
        uint32_t u = uint32_t(r >> 40);
        float f = float(u) * (1.0f / float(1u << 23)) - 1.0f;
        return __float2half(f);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        uint32_t u = uint32_t(r >> 40);
        float f = float(u) * (1.0f / float(1u << 23)) - 1.0f;
        return __float2bfloat16(f);
    } else {
        // integer types: small non-zero values, biased away from zero so div is meaningful
        int v = int(r & 0x7f) + 1;                     // 1..128
        if (r & 0x100) v = -v;                          // sometimes negative
        return T(v);
    }
}

template <typename T>
__global__ void rand_fill_kernel(T* __restrict__ p, int64_t n, uint64_t seed) {
    int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        p[i] = from_random<T>(splitmix64(seed ^ uint64_t(i)));
    }
}

template <typename T>
inline void rand_fill(T* p, int64_t n, uint64_t seed = 0xC0FFEE) {
    int block = 256;
    int64_t nblocks = (n + block - 1) / block;
    int grid = int(nblocks < 65535 ? nblocks : 65535);
    rand_fill_kernel<T><<<grid, block>>>(p, n, seed);
}

} // namespace bench_init
