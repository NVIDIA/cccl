// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

namespace detail
{
// SquadDesc - squad descriptor
//
// A squad is a collection of warps that work together in a warp-specialized
// kernel. A warp-specialized kernel has multiple squads that perform part of
// the computation.
//
// SquadDesc is a host+device constexpr-compatible class that allows describing
// the warp-specialized layout of a kernel.
//
// SquadDesc is constexpr-compatible and can be created on host and device.
struct SquadDesc
{
  int mSquadIdx  = -1;
  int mWarpCount = -1;

  _CCCL_HIDE_FROM_ABI constexpr SquadDesc() = default;
  _CCCL_API constexpr SquadDesc(int squadIdx, int warpCount) noexcept
      : mSquadIdx(squadIdx)
      , mWarpCount(warpCount)
  {}

  [[nodiscard]] _CCCL_API constexpr int warpCount() const noexcept
  {
    return mWarpCount;
  }

  [[nodiscard]] _CCCL_API constexpr int threadCount() const noexcept
  {
    return 32 * warpCount();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const SquadDesc& lhs, const SquadDesc& rhs) noexcept
  {
    return lhs.mSquadIdx == rhs.mSquadIdx;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const SquadDesc& lhs, const SquadDesc& rhs) noexcept
  {
    return lhs.mSquadIdx != rhs.mSquadIdx;
  }
};
// squadCountThreads
//
// Utility function to count the number of threads in an array of squad
// descriptors. It is used to launch a kernel with the correct number of
// threads.
template <int numSquads>
[[nodiscard]] _CCCL_API inline constexpr int squadCountThreads(const SquadDesc (&squads)[numSquads]) noexcept
{
  int sumThreads = 0;
  for (int gi = 0; gi < numSquads; ++gi)
  {
    sumThreads += squads[gi].threadCount();
  }
  return sumThreads;
}
} // namespace detail

CUB_NAMESPACE_END
