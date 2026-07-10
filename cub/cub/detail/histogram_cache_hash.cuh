// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#include <cuda/std/cstdint>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Cache-slot hashing shared by the direct-atomic histogram kernels and the
// benchmark's adversarial input generators. Keeping this in a neutral header
// lets an overlaid-main benchmark use the exact generator without importing
// any optimized kernel implementation.
#ifndef CUB_HISTO_CACHE_HASH_MODE
// 0: low bits, 1: Fibonacci high bits, 2: xor-fold high bits into low.
#  define CUB_HISTO_CACHE_HASH_MODE 1
#endif

CUB_NAMESPACE_BEGIN
namespace detail::histogram
{
inline constexpr ::cuda::std::uint32_t cache_primary_hash_multiplier   = 2654435761u;
inline constexpr ::cuda::std::uint32_t cache_secondary_hash_multiplier = 2246822519u;

// Map a 32-bit multiplicative hash product into [0, cache_mask + 1), where the
// slot count is a power of two and cache_slot_log2 is its base-2 logarithm.
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE int
cache_slot_from_hash(::cuda::std::uint32_t product, int cache_mask, int cache_slot_log2)
{
#if CUB_HISTO_CACHE_HASH_MODE == 1
  (void) cache_mask;
  return static_cast<int>(product >> (32 - cache_slot_log2));
#elif CUB_HISTO_CACHE_HASH_MODE == 2
  (void) cache_slot_log2;
  return static_cast<int>(((product >> 15) ^ product) & static_cast<::cuda::std::uint32_t>(cache_mask));
#else
  (void) cache_slot_log2;
  return static_cast<int>(product & static_cast<::cuda::std::uint32_t>(cache_mask));
#endif
}
} // namespace detail::histogram
CUB_NAMESPACE_END
