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

#include <nv/target> // NV_IF_TARGET

#include <cstdint> // uint8_t

#include <cuda_runtime.h> // __device__, __host__

#include "constantAssert.h" // constantAssert
#include "SpecialRegisters.cuh" // SpecialRegisters

// SkipSync is a tag type that is used to indicate that a SyncHandler.blockInit
// should forgo syncing.
struct SkipSync
{};

struct SyncHandler
{
  static constexpr int mMaxNumResources = 10;
  static constexpr int mMaxNumPhases    = 4;
  static constexpr int mMaxNumStages    = 32;

  // Whether barriers have been initialized.
  bool mHasInitialized = false;

  // Arrays of barrier locations, number of stages, number of owning threads.
  int mNextResourceHandle = 0;
  int mNumStages[mMaxNumResources];
  int mNumPhases[mMaxNumResources];
  int mNumOwningThreads[mMaxNumResources][mMaxNumPhases];
  uint64_t* mPtrBar[mMaxNumResources][mMaxNumPhases];

  __host__ __device__ inline SyncHandler();
  __host__ __device__ inline ~SyncHandler();

  // SyncHandler is a non-copyable, non-movable type. It must be passed by
  // (mutable) reference to be useful.
  SyncHandler(const SyncHandler&)             = delete; // Delete copy constructor
  SyncHandler(SyncHandler&&)                  = delete; // Delete move constructor
  SyncHandler& operator=(const SyncHandler&)  = delete; // Delete copy assignment
  SyncHandler& operator=(const SyncHandler&&) = delete; // Delete move assignment

  // registerResource and registerPhase can be called on host and device.
  __host__ __device__ inline int registerResource(int numStages);
  __host__ __device__ inline void registerPhase(int resourceHandle, int numOwningThreads, uint64_t* ptrBar);

  // clusterInitSync can only be called on device.
  __device__ inline void clusterInitSync(SpecialRegisters sr, SkipSync);
  __device__ inline void clusterInitSync(SpecialRegisters sr);
};

__host__ __device__ inline SyncHandler::SyncHandler()
{
  for (int ri = 0; ri < mMaxNumResources; ++ri)
  {
    mNumStages[ri] = 0;
    mNumPhases[ri] = 0;
    for (int pi = 0; pi < mMaxNumPhases; ++pi)
    {
      mNumOwningThreads[ri][pi] = 0;
      mPtrBar[ri][pi]           = nullptr;
    }
  }
}

__host__ __device__ inline SyncHandler::~SyncHandler()
{
  constantAssert(mHasInitialized, "SyncHandler must have been initialized at end of kernel.");
}

__host__ __device__ inline int SyncHandler::registerResource(int numStages)
{
  constantAssert(!mHasInitialized, "Cannot register resource after SyncHandler has been initialized.");
  // Avoid exceeding the max number of stages
  constantAssert(mNextResourceHandle < mMaxNumResources, "Cannot register more than 10 resources.");

  // Get a handle
  int handle = mNextResourceHandle;
  mNextResourceHandle++;
  // Set the number of stages
  mNumStages[handle] = numStages;

  return handle;
}

__host__ __device__ inline void SyncHandler::registerPhase(int resourceHandle, int numOwningThreads, uint64_t* ptrBar)
{
  constantAssert(!mHasInitialized, "Cannot register phase after SyncHandler has been initialized.");
  constantAssert(resourceHandle < mNextResourceHandle, "Invalid resource handle.");

  // Get phase index:
  int curPhase = mNumPhases[resourceHandle];
  constantAssert(curPhase < mMaxNumPhases, "Cannot register more phases than maximum.");

  mNumOwningThreads[resourceHandle][curPhase] = numOwningThreads;
  mPtrBar[resourceHandle][curPhase]           = ptrBar;

  mNumPhases[resourceHandle]++;
}

__device__ inline void SyncHandler::clusterInitSync(SpecialRegisters sr, SkipSync)
{
  constantAssert(!mHasInitialized, "Cannot initialize SyncHandler twice.");
  mHasInitialized = true;

  // TODO: This could take a Group parameter to split the barrier
  // initialization across groups. Or it could take the number of warps in the
  // block as a parameter.

  // TODO: This could use vectorized mbarrier initialization
  if (sr.warpIdx == 0 && ptx::elect_sync(~0))
  {
    for (int ri = 0; ri < mMaxNumResources; ++ri)
    {
      if (mNextResourceHandle <= ri)
      {
        continue;
      }
      int resNumPhases = mNumPhases[ri];
      int numStages    = mNumStages[ri];

      // We loop over all phases with a conditional on resNumPhases. If we
      // directly loop over only resNumPhases, then the SROA optimization does
      // not kick in and the mPtrBar and mNumOwningThreads arrays are
      // spilled to the stack.
      for (int pi = 0; pi < mMaxNumPhases; ++pi)
      {
        if (pi < resNumPhases)
        {
          uint64_t* ptrBar     = mPtrBar[ri][pi];
          int numOwningThreads = mNumOwningThreads[ri][pi];
          for (int si = 0; si < numStages; ++si)
          {
            ptx::mbarrier_init(&ptrBar[si], numOwningThreads);
          }
        }
      }
    }
  }
}

__device__ inline void SyncHandler::clusterInitSync(SpecialRegisters sr)
{
  clusterInitSync(sr, SkipSync{});
  __cluster_barrier_arrive_relaxed();
  __cluster_barrier_wait();
}
