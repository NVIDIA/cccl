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

#include <cub/device/dispatch/kernels/warpspeed/allocators/SmemAllocator.h>
#include <cub/device/dispatch/kernels/warpspeed/constantAssert.h>
#include <cub/device/dispatch/kernels/warpspeed/squad/SquadDesc.h>
#include <cub/device/dispatch/kernels/warpspeed/SyncHandler.h>

#include <cuda/__ptx/instructions/fence_proxy_async_generic_sync_restrict.h>
#include <cuda/__ptx/instructions/mbarrier_arrive.h>
#include <cuda/__ptx/instructions/mbarrier_arrive_expect_tx.h>
#include <cuda/__ptx/instructions/mbarrier_try_wait_parity.h>
#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

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

  _CCCL_API
  SmemResourceRaw(SyncHandler& syncHandler, void* ptrBase, int sizeBytes, int strideBytes, int stageCount) noexcept
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
  _CCCL_API void addPhase(SyncHandler& syncHandler, uint64_t* ptrBarrier, const SquadDesc (&squads)[numSquads])
  {
    int numOwningThreads = squadCountThreads(squads);

    int curPhase = mNumPhases;
    mNumPhases++;

    syncHandler.registerPhase(mResourceHandle, numOwningThreads, ptrBarrier);
    mPtrBar[curPhase] = ptrBarrier;
  }

  template <int numSquads>
  _CCCL_API void addPhase(SyncHandler& syncHandler, SmemAllocator& smemAllocator, const SquadDesc (&squads)[numSquads])
  {
    uint64_t* ptrBar =
      reinterpret_cast<uint64_t*>(smemAllocator.alloc(mStageCount * sizeof(uint64_t), alignof(uint64_t)));
    addPhase(syncHandler, ptrBar, squads);
  }

  _CCCL_API void addPhase(SyncHandler& syncHandler, uint64_t* ptrBarrier, const SquadDesc& squad)
  {
    const SquadDesc squads[] = {squad};
    addPhase(syncHandler, ptrBarrier, squads);
  }

  _CCCL_API void addPhase(SyncHandler& syncHandler, SmemAllocator& smemAllocator, const SquadDesc& squad)
  {
    const SquadDesc squads[] = {squad};
    addPhase(syncHandler, smemAllocator, squads);
  }

  _CCCL_DEVICE_API void incrementStage()
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

  [[nodiscard]] _CCCL_DEVICE_API void* data()
  {
    return (void*) (mPtrBase + mStageCurrent * mStride);
  }

  [[nodiscard]] _CCCL_DEVICE_API uint64_t* ptrCurBarrierRelease(int phase)
  {
    uint64_t* ptrBarPhase = mPtrBar[phase];
    constantAssert(phase < mNumPhases, "Phase exceeds limit.");
    return &ptrBarPhase[mStageCurrent];
  }
  _CCCL_DEVICE_API void release(int phase)
  {
    ::cuda::ptx::mbarrier_arrive(ptrCurBarrierRelease(phase));
  }

  _CCCL_DEVICE_API void releaseTx(int phase, int txCount)
  {
    ::cuda::ptx::mbarrier_arrive_expect_tx(
      ::cuda::ptx::sem_release, ::cuda::ptx::scope_cta, ::cuda::ptx::space_shared, ptrCurBarrierRelease(phase), txCount);
  }

  _CCCL_DEVICE_API void fenceLdsToAsyncProxy()
  {
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
  }

  _CCCL_DEVICE_API void releaseLdsToAsyncProxy(int phase)
  {
    // First fence
    fenceLdsToAsyncProxy();
    // Then perform a normal release
    release(phase);
  }

  _CCCL_DEVICE_API void acquire(int phase)
  {
    constantAssert(phase < mNumPhases, "Phase exceeds limit.");

    // The release of the previous phase occurs on the `phase - 1` barrier. So
    // that is what we wait on.
    int phaseAcq          = (mNumPhases + phase - 1) % mNumPhases;
    uint64_t* ptrBarPhase = mPtrBar[phaseAcq];

    while (!::cuda::ptx::mbarrier_try_wait_parity(&ptrBarPhase[mStageCurrent], mParity[phase]))
    {
    }
  }
};

CUB_NAMESPACE_END
