/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cuda/ptx> // ptx::elect_sync()

#include <cuda_runtime.h> // __host__, __device__

#include "../SpecialRegisters.cuh" // SpecialRegisters
#include "SquadDesc.h" // SquadDesc

namespace ptx = cuda::ptx;
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

  __device__ inline Squad(SquadDesc squadStatic, SpecialRegisters specialRegisters);

  __device__ inline int warpRank() const;
  __device__ inline int threadRank() const;
  __device__ inline bool isLeaderThread() const;
  __device__ inline bool isLeaderWarp() const;
  __device__ inline bool isLeaderThreadOfWarp() const;
  __device__ inline void syncThreads();
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
template <int numSquads, typename F>
__device__ inline void
squadDispatch(SpecialRegisters sr, const SquadDesc (&squads)[numSquads], F f, int warpIdxStart = 0);
// Squad
__device__ inline Squad::Squad(SquadDesc squadStatic, SpecialRegisters specialRegisters)
    : SquadDesc(squadStatic)
    , mSpecialRegisters(specialRegisters)
{
  mIsWarpLeader = ptx::elect_sync(~0);
  mIsLeaderWarp = warpRank() == 0;
}

__device__ inline int Squad::warpRank() const
{
  return mSpecialRegisters.warpIdx % this->warpCount();
}

__device__ inline int Squad::threadRank() const
{
  return mSpecialRegisters.threadIdxX % this->threadCount();
}

__device__ inline bool Squad::isLeaderThread() const
{
  return mIsWarpLeader && mIsLeaderWarp;
}
__device__ inline bool Squad::isLeaderWarp() const
{
  return mIsLeaderWarp;
}

__device__ inline bool Squad::isLeaderThreadOfWarp() const
{
  return mIsWarpLeader;
}

__device__ inline void Squad::syncThreads()
{
  // barrier 0 is reserved for __syncthreads(). We use barrier ids 1, ...
  int barrierIdx = (int) this->mSquadIdx + 1;

  __barrier_sync_count(barrierIdx, this->threadCount());
}
// squadDispatch
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
__device__ inline void squadDispatch(SpecialRegisters sr, const SquadDesc (&squads)[numSquads], F f, int warpIdxStart)
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
