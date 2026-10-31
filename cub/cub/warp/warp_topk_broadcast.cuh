// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cmath>

CUB_NAMESPACE_BEGIN
namespace detail
{
//! Experimental register top-k with sorted results replicated to every lane.
//! Input length, K, and LogicalWarpThreads are powers of two; K and LogicalWarpThreads
//! are no greater than 32. Each contiguous logical warp selects its own keys.
//! Every lane of the complete physical warp must be active and call the same
//! invocation; independently divergent logical warps and partial warps are unsupported.
//! Ties are interchangeable; callers can encode index tie-breaking in integer keys.
//! When per-thread input is shorter than K, the caller must supply a sentinel no
//! greater than any input key. Negative infinity is valid for floating-point keys.
//! Floating-point NaNs are outside the experimental contract.
//! Float32 and integral keys are supported; this experimental form selects the largest keys.
//! Input arrays are preserved and may contain any partition of the warp input.
//! Input and output arrays must not overlap.
//! Output arrays within each logical warp contain the same K values in descending order.
template <typename KeyT, int K, int LogicalWarpThreads = 32>
class WarpTopKBroadcast
{
  static_assert(K > 0 && K <= 32 && (K & (K - 1)) == 0);
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= 32
                && (LogicalWarpThreads & (LogicalWarpThreads - 1)) == 0);
  static_assert(::cuda::std::is_integral_v<KeyT> || ::cuda::std::is_same_v<KeyT, float>);

  [[nodiscard]] _CCCL_DEVICE_API static KeyT maximum(KeyT a, KeyT b) noexcept
  {
    if constexpr (::cuda::std::is_same_v<KeyT, float>)
    {
      return ::fmaxf(a, b);
    }
    else
    {
      return a < b ? b : a;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static KeyT minimum(KeyT a, KeyT b) noexcept
  {
    if constexpr (::cuda::std::is_same_v<KeyT, float>)
    {
      return ::fminf(a, b);
    }
    else
    {
      return a < b ? a : b;
    }
  }

  template <int Count>
  _CCCL_DEVICE_API static void merge(KeyT (&keys)[Count]) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int step = Count / 2; step > 0; step /= 2)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Count; ++i)
      {
        if ((i & step) == 0)
        {
          const KeyT a   = keys[i];
          const KeyT b   = keys[i + step];
          keys[i]        = maximum(a, b);
          keys[i + step] = minimum(a, b);
        }
      }
    }
  }

  _CCCL_DEVICE_API static void compare_exchange(KeyT& a, KeyT& b) noexcept
  {
    const KeyT high = maximum(a, b);
    const KeyT low  = minimum(a, b);
    a               = high;
    b               = low;
  }

  template <int Count>
  _CCCL_DEVICE_API static void sort(KeyT (&keys)[Count]) noexcept
  {
    // Small sorting networks from Bert Dobbelaere's public network list:
    // https://bertdobbelaere.github.io/sorting_networks.html
    // These match QuACK's float32 networks. Integral keys retain bitonic sorting.
    if constexpr (::cuda::std::is_same_v<KeyT, float> && Count == 4)
    {
      compare_exchange(keys[0], keys[2]);
      compare_exchange(keys[1], keys[3]);

      compare_exchange(keys[0], keys[1]);
      compare_exchange(keys[2], keys[3]);

      compare_exchange(keys[1], keys[2]);
    }
    else if constexpr (::cuda::std::is_same_v<KeyT, float> && Count == 8)
    {
      compare_exchange(keys[0], keys[2]);
      compare_exchange(keys[1], keys[3]);
      compare_exchange(keys[4], keys[6]);
      compare_exchange(keys[5], keys[7]);

      compare_exchange(keys[0], keys[4]);
      compare_exchange(keys[1], keys[5]);
      compare_exchange(keys[2], keys[6]);
      compare_exchange(keys[3], keys[7]);

      compare_exchange(keys[0], keys[1]);
      compare_exchange(keys[2], keys[3]);
      compare_exchange(keys[4], keys[5]);
      compare_exchange(keys[6], keys[7]);

      compare_exchange(keys[2], keys[4]);
      compare_exchange(keys[3], keys[5]);

      compare_exchange(keys[1], keys[4]);
      compare_exchange(keys[3], keys[6]);

      compare_exchange(keys[1], keys[2]);
      compare_exchange(keys[3], keys[4]);
      compare_exchange(keys[5], keys[6]);
    }
    else if constexpr (::cuda::std::is_same_v<KeyT, float> && Count == 16)
    {
      compare_exchange(keys[0], keys[13]);
      compare_exchange(keys[1], keys[12]);
      compare_exchange(keys[2], keys[15]);
      compare_exchange(keys[3], keys[14]);
      compare_exchange(keys[4], keys[8]);
      compare_exchange(keys[5], keys[6]);
      compare_exchange(keys[7], keys[11]);
      compare_exchange(keys[9], keys[10]);

      compare_exchange(keys[0], keys[5]);
      compare_exchange(keys[1], keys[7]);
      compare_exchange(keys[2], keys[9]);
      compare_exchange(keys[3], keys[4]);
      compare_exchange(keys[6], keys[13]);
      compare_exchange(keys[8], keys[14]);
      compare_exchange(keys[10], keys[15]);
      compare_exchange(keys[11], keys[12]);

      compare_exchange(keys[0], keys[1]);
      compare_exchange(keys[2], keys[3]);
      compare_exchange(keys[4], keys[5]);
      compare_exchange(keys[6], keys[8]);
      compare_exchange(keys[7], keys[9]);
      compare_exchange(keys[10], keys[11]);
      compare_exchange(keys[12], keys[13]);
      compare_exchange(keys[14], keys[15]);

      compare_exchange(keys[0], keys[2]);
      compare_exchange(keys[1], keys[3]);
      compare_exchange(keys[4], keys[10]);
      compare_exchange(keys[5], keys[11]);
      compare_exchange(keys[6], keys[7]);
      compare_exchange(keys[8], keys[9]);
      compare_exchange(keys[12], keys[14]);
      compare_exchange(keys[13], keys[15]);

      compare_exchange(keys[1], keys[2]);
      compare_exchange(keys[3], keys[12]);
      compare_exchange(keys[4], keys[6]);
      compare_exchange(keys[5], keys[7]);
      compare_exchange(keys[8], keys[10]);
      compare_exchange(keys[9], keys[11]);
      compare_exchange(keys[13], keys[14]);

      compare_exchange(keys[1], keys[4]);
      compare_exchange(keys[2], keys[6]);
      compare_exchange(keys[5], keys[8]);
      compare_exchange(keys[7], keys[10]);
      compare_exchange(keys[9], keys[13]);
      compare_exchange(keys[11], keys[14]);

      compare_exchange(keys[2], keys[4]);
      compare_exchange(keys[3], keys[6]);
      compare_exchange(keys[9], keys[12]);
      compare_exchange(keys[11], keys[13]);

      compare_exchange(keys[3], keys[5]);
      compare_exchange(keys[6], keys[8]);
      compare_exchange(keys[7], keys[9]);
      compare_exchange(keys[10], keys[12]);

      compare_exchange(keys[3], keys[4]);
      compare_exchange(keys[5], keys[6]);
      compare_exchange(keys[7], keys[8]);
      compare_exchange(keys[9], keys[10]);
      compare_exchange(keys[11], keys[12]);

      compare_exchange(keys[6], keys[7]);
      compare_exchange(keys[8], keys[9]);
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int size = 2; size <= Count; size *= 2)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int step = size / 2; step > 0; step /= 2)
        {
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int i = 0; i < Count; ++i)
          {
            if ((i & step) == 0)
            {
              const KeyT a    = keys[i];
              const KeyT b    = keys[i + step];
              const KeyT high = maximum(a, b);
              const KeyT low  = minimum(a, b);
              keys[i]         = (i & size) == 0 ? high : low;
              keys[i + step]  = (i & size) == 0 ? low : high;
            }
          }
        }
      }
    }
  }

  _CCCL_DEVICE_API static void select(KeyT (&keys)[K], const KeyT (&other)[K]) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < K; ++i)
    {
      keys[i] = maximum(keys[i], other[K - 1 - i]);
    }
    merge(keys);
  }

public:
  template <int ItemsPerThread>
  _CCCL_DEVICE_API static void TopK(const KeyT (&input)[ItemsPerThread], KeyT (&output)[K], KeyT sentinel) noexcept
  {
    static_assert(ItemsPerThread > 0 && (ItemsPerThread & (ItemsPerThread - 1)) == 0);
    static_assert(ItemsPerThread * LogicalWarpThreads >= K);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < K; ++i)
    {
      output[i] = i < ItemsPerThread ? input[i] : sentinel;
    }
    sort(output);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int chunk = 1; chunk < ItemsPerThread / K; ++chunk)
    {
      KeyT other[K];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < K; ++i)
      {
        other[i] = input[chunk * K + i];
      }
      sort(other);
      select(output, other);
    }
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = 1; offset < LogicalWarpThreads; offset *= 2)
    {
      KeyT other[K];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < K; ++i)
      {
        other[i] = ::cuda::device::warp_shuffle_xor<LogicalWarpThreads>(output[i], offset);
      }
      select(output, other);
    }
  }
};
} // namespace detail
CUB_NAMESPACE_END
