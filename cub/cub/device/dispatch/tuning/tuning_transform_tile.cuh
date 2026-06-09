// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Policy picker for cub::DeviceTransform's tile path. Mirrors the
// bytes-in-flight target used by CUB's non-tile algorithms (see
// tuning_transform.cuh's cc_to_min_bytes_in_flight) but expresses the
// answer as a TileSize, since tile kernels partition by compile-time
// shape rather than threads*items.

#pragma once

#include <cub/config.cuh>

#include <cub/device/dispatch/dispatch_transform_tile_config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUB_HAS_TILE_TRANSFORM()

#  include <cuda/cmath>

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{

constexpr int min_bytes_in_flight_per_sm(int cc_x10)
{
  if (cc_x10 >= 1000)
  {
    return 64 * 1024; // B200
  }
  if (cc_x10 >= 900)
  {
    return 48 * 1024; // H100/H200
  }
  if (cc_x10 >= 800)
  {
    return 16 * 1024; // A100
  }
  return 12 * 1024;
}

constexpr int min_size(int a)
{
  return a;
}
template <class... Ts>
constexpr int min_size(int a, int b, Ts... rest)
{
  int m = a < b ? a : b;
  return min_size(m, rest...);
}

// mufu_heavy=true tells the policy the functor body has heavy MUFU usage.
// for small data types, vectorized load will make them arrive packed in
// registers and the compiler unpacks them and packs them back. reducing the
// compute work per thread helps here. need profiling to know the exact cause.
template <typename Out, typename... Ins>
constexpr int pick_tile_size(bool mufu_heavy = false, int cc_x10 = 1000)
{
  constexpr int threads_per_block    = 128;
  constexpr int vector_bytes         = 16; // LDG.E.128 -> 16 bytes
  constexpr int max_items_per_thread = 32;
  constexpr int max_occupancy        = 16;

  constexpr int min_elem      = min_size(int(sizeof(Out)), int(sizeof(Ins))...);
  constexpr int items_for_vec = static_cast<int>(::cuda::ceil_div(vector_bytes, min_elem));

  // Fill (zero inputs) keeps the same latency target by counting output bytes.
  constexpr int bytes_per_iter = (sizeof...(Ins) > 0) ? (int(sizeof(Ins)) + ... + 0) : int(sizeof(Out));
  const int target             = min_bytes_in_flight_per_sm(cc_x10);
  const int items_for_latency =
    static_cast<int>(::cuda::ceil_div(target, max_occupancy * threads_per_block * bytes_per_iter));

  int items = items_for_vec > items_for_latency ? items_for_vec : items_for_latency;
  items     = static_cast<int>(::cuda::next_power_of_two(static_cast<unsigned int>(items)));
  if (items > max_items_per_thread)
  {
    items = max_items_per_thread;
  }

  if (mufu_heavy && min_elem < 4)
  {
    const int byte_cap = vector_bytes / min_elem; // 16 for I8, 8 for I16/half/bf16
    if (items > byte_cap)
    {
      items = byte_cap;
    }
  }

  return items * threads_per_block;
}

} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CUB_HAS_TILE_TRANSFORM()
