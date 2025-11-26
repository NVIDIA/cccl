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

#include <cstdint> // uint8_t

#include "../allocators/SmemAllocator.h" // SmemAllocator
#include "../constantAssert.h" // constantAssert
#include "../squad/SquadDesc.h" // SquadDesc
#include "../SyncHandler.h" // SyncHandler

struct SmemResourceRaw
{
  static constexpr int mMaxNumPhases = 4;

  int mStageCurrent = 0;

  int mResourceHandle;
  uint8_t* mPtrBase;
  int mSizeBytes;
  int mStride;
  int mStageCount;
  int mNumPhases;

  uint64_t* mPtrBar[mMaxNumPhases];
  int mParity[mMaxNumPhases];

  __host__ __device__ inline SmemResourceRaw(
    SyncHandler& syncHandler, void* ptrBase, int sizeBytes, int strideBytes, int stageCount);

  template <int numSquads>
  __host__ __device__ inline void
  addPhase(SyncHandler& syncHandler, uint64_t* ptrBarrier, const SquadDesc (&squads)[numSquads]);
  template <int numSquads>
  __host__ __device__ inline void
  addPhase(SyncHandler& syncHandler, SmemAllocator& smemAllocator, const SquadDesc (&squads)[numSquads]);

  __host__ __device__ inline void addPhase(SyncHandler& syncHandler, uint64_t* ptrBarrier, const SquadDesc& squad);
  __host__ __device__ inline void
  addPhase(SyncHandler& syncHandler, SmemAllocator& smemAllocator, const SquadDesc& squad);

  __device__ inline void incrementStage();
  __device__ inline void* data();

  __device__ inline uint64_t* ptrCurBarrierRelease(int phase);
  __device__ inline void release(int phase);
  __device__ inline void releaseTx(int phase, int txCount);
  __device__ inline void fenceLdsToAsyncProxy();
  __device__ inline void releaseLdsToAsyncProxy(int phase);
  __device__ inline void acquire(int phase);
};

__host__ __device__ inline SmemResourceRaw::SmemResourceRaw(
  SyncHandler& syncHandler, void* ptrBase, int sizeBytes, int strideBytes, int stageCount)
    : mResourceHandle(syncHandler.registerResource(stageCount))
    , mPtrBase((uint8_t*) ptrBase)
    , mSizeBytes(sizeBytes)
    , mStride(strideBytes)
    , mStageCount(stageCount)
    , mNumPhases(0)
{
  for (int pi = 0; pi < mMaxNumPhases; ++pi)
  {
    mPtrBar[pi] = nullptr;
    mParity[pi] = pi == 0 ? 1 : 0;
  }
}

template <int numSquads>
__host__ __device__ inline void
SmemResourceRaw::addPhase(SyncHandler& syncHandler, uint64_t* ptrBarrier, const SquadDesc (&squads)[numSquads])
{
  int numOwningThreads = squadCountThreads(squads);

  int curPhase = mNumPhases;
  mNumPhases++;

  syncHandler.registerPhase(mResourceHandle, numOwningThreads, ptrBarrier);
  mPtrBar[curPhase] = ptrBarrier;
}

template <int numSquads>
__host__ __device__ inline void
SmemResourceRaw::addPhase(SyncHandler& syncHandler, SmemAllocator& smemAllocator, const SquadDesc (&squads)[numSquads])
{
  uint64_t* ptrBar =
    reinterpret_cast<uint64_t*>(smemAllocator.alloc(mStageCount * sizeof(uint64_t), alignof(uint64_t)));
  addPhase(syncHandler, ptrBar, squads);
}

__host__ __device__ inline void
SmemResourceRaw::addPhase(SyncHandler& syncHandler, uint64_t* ptrBarrier, const SquadDesc& squad)
{
  const SquadDesc squads[] = {squad};
  addPhase(syncHandler, ptrBarrier, squads);
}

__host__ __device__ inline void
SmemResourceRaw::addPhase(SyncHandler& syncHandler, SmemAllocator& smemAllocator, const SquadDesc& squad)
{
  const SquadDesc squads[] = {squad};
  addPhase(syncHandler, smemAllocator, squads);
}

__device__ inline void SmemResourceRaw::incrementStage()
{
  if (mStageCurrent == mStageCount - 1)
  {
    mStageCurrent = 0;
    // We loop over all phases with a conditional on resNumPhases. If we
    // directly loop over only resNumPhases, then the SROA optimization does
    // not kick in and the mParity array is spilled to the stack.
    for (int pi = 0; pi < mMaxNumPhases; ++pi)
    {
      if (pi < mNumPhases)
      {
        mParity[pi] ^= 1;
      }
    }
  }
  else
  {
    mStageCurrent++;
  }
}

__device__ inline void* SmemResourceRaw::data()
{
  return (void*) (mPtrBase + mStageCurrent * mStride);
}

__device__ inline uint64_t* SmemResourceRaw::ptrCurBarrierRelease(int phase)
{
  uint64_t* ptrBarPhase = mPtrBar[phase];
  constantAssert(phase < mNumPhases, "Phase exceeds limit.");
  return &ptrBarPhase[mStageCurrent];
}

__device__ inline void SmemResourceRaw::release(int phase)
{
  ptx::mbarrier_arrive(ptrCurBarrierRelease(phase));
}

__device__ inline void SmemResourceRaw::releaseTx(int phase, int txCount)
{
  ptx::mbarrier_arrive_expect_tx(
    ptx::sem_release, ptx::scope_cta, ptx::space_shared, ptrCurBarrierRelease(phase), txCount);
}

__device__ inline void SmemResourceRaw::fenceLdsToAsyncProxy()
{
  ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
}

__device__ inline void SmemResourceRaw::releaseLdsToAsyncProxy(int phase)
{
  // First fence
  fenceLdsToAsyncProxy();
  // Then perform a normal release
  release(phase);
}

__device__ inline void SmemResourceRaw::acquire(int phase)
{
  constantAssert(phase < mNumPhases, "Phase exceeds limit.");

  // The release of the previous phase occurs on the `phase - 1` barrier. So
  // that is what we wait on.
  int phaseAcq          = (mNumPhases + phase - 1) % mNumPhases;
  uint64_t* ptrBarPhase = mPtrBar[phaseAcq];

  while (!ptx::mbarrier_try_wait_parity(&ptrBarPhase[mStageCurrent], mParity[phase]))
  {
  }
}
