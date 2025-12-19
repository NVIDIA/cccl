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

#include <cub/device/dispatch/kernels/warpspeed/SpecialRegisters.cuh>
#include <cub/device/dispatch/kernels/warpspeed/squad/SquadDesc.h>

#include <cuda/__ptx/instructions/elect_sync.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
// Squad - device squad instance
//
// A squad is a collection of warps that work together in a warp-specialized
// kernel. A warp-specialized kernel has multiple squads that perform part of
// the computation.
//
// The Squad class is a device runtime instance of a squad. It provides
// functionality to determine the rank of the current thread or warp in the
// squad, and to sync all threads in the squad.
struct Squad : SquadDesc
{
  SpecialRegisters mSpecialRegisters;
  bool mIsWarpLeader = false;
  bool mIsLeaderWarp = false;

  _CCCL_DEVICE_API Squad(SquadDesc squadStatic, SpecialRegisters specialRegisters)
      : SquadDesc(squadStatic)
      , mSpecialRegisters(specialRegisters)
  {
    mIsWarpLeader = ::cuda::ptx::elect_sync(~0);
    mIsLeaderWarp = warpRank() == 0;
  }

  [[nodiscard]] _CCCL_DEVICE_API int warpRank() const
  {
    return mSpecialRegisters.warpIdx % this->warpCount();
  }

  [[nodiscard]] _CCCL_DEVICE_API int threadRank() const
  {
    return mSpecialRegisters.threadIdxX % this->threadCount();
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderThread() const
  {
    return mIsWarpLeader && mIsLeaderWarp;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderWarp() const
  {
    return mIsLeaderWarp;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool isLeaderThreadOfWarp() const
  {
    return mIsWarpLeader;
  }

  _CCCL_DEVICE_API void syncThreads()
  {
    // barrier 0 is reserved for __syncthreads(). We use barrier ids 1, ...
    int barrierIdx = (int) this->mSquadIdx + 1;

    __barrier_sync_count(barrierIdx, this->threadCount());
  }
};
// squadDispatch
//
// squadDispatch is used at the start of the kernel. It takes an array of squad
// descriptors and determines which squad the current thread belongs to. The
// lambda `f: (Squad) -> void` is called with the squad currently active on this
// thread.
//
// Typically, the user will call the kernel body with the active squad.
//
// Implementation notes:
//
// Dispatch to squad based on warp index using a binary search. This balances
// the number of BRA instructions per squad and avoids NVVM inserting BRX
// instructions. BRX instructions require a jump table that is loaded from GCC,
// which incurs latency.
//
// The benefit of this function for fastScan is that adding a new squad doesn't
// require code changes in the dispatch. For low-latency inference, I hope that
// the avoidance of linear search and BRX instructions translates into latency
// reductions.
//
template <int numSquads, typename F>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
squadDispatch(SpecialRegisters sr, const SquadDesc (&squads)[numSquads], F f, int warpIdxStart = 0)
{
  static_assert(numSquads > 0);
  if (numSquads == 1)
  {
    // Leaf
    SquadDesc squad = squads[0];

    if (warpIdxStart <= sr.warpIdx && sr.warpIdx < warpIdxStart + squad.warpCount())
    {
      f(Squad(squad, sr));
    }
  }
  else
  {
    constexpr int mid = numSquads / 2;
    // Left
    int warpIdxStartMid = warpIdxStart;
    for (int gi = 0; gi < mid; ++gi)
    {
      warpIdxStartMid += squads[gi].warpCount();
    }
    if (sr.warpIdx < warpIdxStartMid)
    {
      if constexpr (0 < mid)
      {
        SquadDesc squadsLeft[mid];
        for (int gi = 0; gi < mid; ++gi)
        {
          squadsLeft[gi] = squads[gi];
        }
        squadDispatch(sr, squadsLeft, f, warpIdxStart);
      }
    }
    else
    {
      SquadDesc squadsRight[numSquads - mid]{};
      for (int gi = 0; gi < numSquads - mid; ++gi)
      {
        squadsRight[gi] = squads[mid + gi];
      }
      squadDispatch(sr, squadsRight, f, warpIdxStartMid);
    }
  }
}
} // namespace detail::scan

CUB_NAMESPACE_END
