// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cuda_runtime.h> // __host__, __device__
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

  __host__ __device__ inline constexpr SquadDesc(int squadIdx, int warpCount);

  // SquadDesc is a default-constructible, copyable, and movable type.
  inline constexpr SquadDesc() = default;
  //__host__ __device__ inline constexpr SquadDesc(const SquadDesc& other) = default;
  //__host__ __device__ inline constexpr SquadDesc& operator=(const SquadDesc& other) = default;
  //__host__ __device__ inline constexpr SquadDesc(SquadDesc&& other) = default;
  //__host__ __device__ inline constexpr SquadDesc& operator=(SquadDesc&& other) = default;

  __host__ __device__ inline constexpr int warpCount() const;
  __host__ __device__ inline constexpr int threadCount() const;
  __host__ __device__ inline bool operator==(const SquadDesc& other) const;
};
// squadCountThreads
//
// Utility function to count the number of threads in an array of squad
// descriptors. It is used to launch a kernel with the correct number of
// threads.
template <int numSquads>
__host__ __device__ inline constexpr int squadCountThreads(const SquadDesc (&squads)[numSquads]);
// SquadDesc
__host__ __device__ inline constexpr SquadDesc::SquadDesc(int squadIdx, int warpCount)
    : mSquadIdx(squadIdx)
    , mWarpCount(warpCount)
{}

__host__ __device__ inline constexpr int SquadDesc::warpCount() const
{
  return mWarpCount;
}

__host__ __device__ inline constexpr int SquadDesc::threadCount() const
{
  return 32 * warpCount();
}

__host__ __device__ inline bool SquadDesc::operator==(const SquadDesc& other) const
{
  return this->mSquadIdx == other.mSquadIdx;
}
// squadCountThreads
//
template <int numSquads>
__host__ __device__ inline constexpr int squadCountThreads(const SquadDesc (&squads)[numSquads])
{
  int sumThreads = 0;
  for (int gi = 0; gi < numSquads; ++gi)
  {
    sumThreads += squads[gi].threadCount();
  }
  return sumThreads;
}
