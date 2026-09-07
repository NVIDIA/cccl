// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Policy picker for cub::DeviceTransform's tile path. Shares the
// bytes-in-flight target used by CUB's non-tile algorithms (calls
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

#  include <cub/device/dispatch/tuning/tuning_transform.cuh>

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__cmath/pow2.h>
#  include <cuda/__device/compute_capability.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__cstddef/types.h>

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{
// mufu_heavy=true tells the policy the functor body has heavy MUFU usage.
// for small data types, vectorized load will make them arrive packed in
// registers and the compiler unpacks them and packs them back. reducing the
// compute work per thread helps here. need profiling to know the exact cause.
template <typename Out, typename... Ins>
constexpr int pick_tile_size(bool mufu_heavy = false, ::cuda::compute_capability cc = {10, 0})
{
  constexpr int threads_per_block    = 128;
  constexpr int vector_bytes         = 16; // LDG.E.128 -> 16 bytes
  constexpr int max_items_per_thread = 32;
  constexpr int max_occupancy        = 16;

  constexpr auto min_elem     = ::cuda::std::min({sizeof(Out), sizeof(Ins)...});
  constexpr int items_for_vec = ::cuda::ceil_div(vector_bytes, min_elem);

  // Fill (zero inputs) keeps the same latency target by counting output bytes.
  constexpr auto bytes_per_iter = (sizeof...(Ins) > 0) ? (sizeof(Ins) + ... + ::cuda::std::size_t{0}) : sizeof(Out);
  const int target              = cub::detail::transform::cc_to_min_bytes_in_flight(cc);
  const int items_for_latency =
    static_cast<int>(::cuda::ceil_div(target, max_occupancy * threads_per_block * bytes_per_iter));

  int items = ::cuda::std::max(items_for_vec, items_for_latency);
  items     = static_cast<int>(::cuda::next_power_of_two(static_cast<unsigned>(items)));
  if (items > max_items_per_thread)
  {
    items = max_items_per_thread;
  }

  if (mufu_heavy && min_elem < 4)
  {
    // Elements that fit in one 16-byte vector load -> items/thread cap for MUFU-heavy sub-4B ops.
    // min_elem is size_t, so cast the quotient once here to keep this an int item count (matches
    // items below, so the comparison/assignment stay int-vs-int: no sign-compare, no use-site casts).
    constexpr int vec_items_cap = static_cast<int>(vector_bytes / min_elem); // 16 for I8, 8 for I16/half/bf16
    if (items > vec_items_cap)
    {
      items = vec_items_cap;
    }
  }

  return items * threads_per_block;
}
} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CUB_HAS_TILE_TRANSFORM()
