// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/util_ptx.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
//! Experimental float32 sum with a result in every participating logical lane.
//!
//! LogicalWarpThreads must be a power of two in [1, 32]. Logical warps are
//! contiguous aligned groups within a physical warp. Every lane of a logical
//! warp must participate; partial logical warps are unsupported.
//!
//! With SyncPhysicalWarp=false, logical warps may participate independently,
//! including from divergent branches. With SyncPhysicalWarp=true, all 32 lanes
//! of the physical warp must execute the same invocation together. This option
//! can reduce shuffle synchronization overhead for converged consumers.
//!
//! Sum uses an increasing-XOR butterfly and permits floating-point
//! reassociation. It does not preserve WarpReduce::Sum's evaluation order or
//! promise bitwise equality with it. Inputs and intermediate sums must be
//! finite. No temporary storage is required.
template <typename T, int LogicalWarpThreads = 32, bool SyncPhysicalWarp = false>
class WarpAllReduce
{
  static_assert(::cuda::std::is_same_v<T, float>, "The experimental implementation supports float32");
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= 32
                  && (LogicalWarpThreads & (LogicalWarpThreads - 1)) == 0,
                "LogicalWarpThreads must be a power of two in [1, 32]");

public:
  //! Sum one value from each logical lane and return the sum in every lane.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T value) const noexcept
  {
    const unsigned mask =
      SyncPhysicalWarp ? 0xffffffffu : cub::WarpMask<LogicalWarpThreads>(detail::logical_warp_id<LogicalWarpThreads>());
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = 1; offset < LogicalWarpThreads; offset *= 2)
    {
      value += ::cuda::device::warp_shuffle_xor<LogicalWarpThreads>(value, offset, mask);
    }
    return value;
  }
};
} // namespace detail
CUB_NAMESPACE_END
