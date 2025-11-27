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

  _CCCL_API constexpr SquadDesc(int squadIdx, int warpCount);

  // SquadDesc is a default-constructible, copyable, and movable type.
  inline constexpr SquadDesc() = default;
  //_CCCL_API constexpr SquadDesc(const SquadDesc& other) = default;
  //_CCCL_API constexpr SquadDesc& operator=(const SquadDesc& other) = default;
  //_CCCL_API constexpr SquadDesc(SquadDesc&& other) = default;
  //_CCCL_API constexpr SquadDesc& operator=(SquadDesc&& other) = default;

  _CCCL_API constexpr int warpCount() const;
  _CCCL_API constexpr int threadCount() const;
  _CCCL_API bool operator==(const SquadDesc& other) const;
};
// squadCountThreads
//
// Utility function to count the number of threads in an array of squad
// descriptors. It is used to launch a kernel with the correct number of
// threads.
template <int numSquads>
_CCCL_API constexpr int squadCountThreads(const SquadDesc (&squads)[numSquads]);
// SquadDesc
_CCCL_API constexpr SquadDesc::SquadDesc(int squadIdx, int warpCount)
    : mSquadIdx(squadIdx)
    , mWarpCount(warpCount)
{}

_CCCL_API constexpr int SquadDesc::warpCount() const
{
  return mWarpCount;
}

_CCCL_API constexpr int SquadDesc::threadCount() const
{
  return 32 * warpCount();
}

_CCCL_API bool SquadDesc::operator==(const SquadDesc& other) const
{
  return this->mSquadIdx == other.mSquadIdx;
}
// squadCountThreads
//
template <int numSquads>
_CCCL_API constexpr int squadCountThreads(const SquadDesc (&squads)[numSquads])
{
  int sumThreads = 0;
  for (int gi = 0; gi < numSquads; ++gi)
  {
    sumThreads += squads[gi].threadCount();
  }
  return sumThreads;
}
