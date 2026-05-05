// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/constant_assert.cuh>
#include <cub/detail/warpspeed/special_registers.cuh>

#include <cuda/__ptx/instructions/mbarrier_init.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
// SkipSync is a tag type that is used to indicate that a SyncHandler.blockInit
// should forgo syncing.
struct SkipSync
{};

struct SyncHandler
{
  // reducing these values to the actually used number of resources and phases does not improve performance
  static constexpr int mMaxNumResources = 10;
  static constexpr int mMaxNumPhases    = 4;

  // Whether barriers have been initialized.
  bool mHasInitialized = false;

  // Arrays of barrier locations, number of stages, number of owning threads.
  int mNextResourceHandle = 0;
  int mNumStages[mMaxNumResources]{};
  int mNumPhases[mMaxNumResources]{};
  int mNumOwningThreads[mMaxNumResources][mMaxNumPhases]{};
  ::cuda::std::uint64_t* mPtrBar[mMaxNumResources][mMaxNumPhases]{};

  constexpr SyncHandler() = default;

  // we need constant destruction for the host side single stage SMEM amount, which is only possible in C++20
#if _CCCL_STD_VER >= 2020
  _CCCL_API constexpr ~SyncHandler()
  {
    _WS_CONSTANT_ASSERT(mHasInitialized, "SyncHandler must have been initialized at end of kernel.");
  }
#endif // _CCCL_STD_VER >= 2020

  // SyncHandler is a non-copyable, non-movable type. It must be passed by
  // (mutable) reference to be useful.
  SyncHandler(const SyncHandler&)             = delete; // Delete copy constructor
  SyncHandler(SyncHandler&&)                  = delete; // Delete move constructor
  SyncHandler& operator=(const SyncHandler&)  = delete; // Delete copy assignment
  SyncHandler& operator=(const SyncHandler&&) = delete; // Delete move assignment

  // registerResource and registerPhase can be called on host and device.
  [[nodiscard]] _CCCL_API constexpr int registerResource(int numStages)
  {
    _WS_CONSTANT_ASSERT(!mHasInitialized, "Cannot register resource after SyncHandler has been initialized.");
    // Avoid exceeding the max number of stages
    _WS_CONSTANT_ASSERT(mNextResourceHandle < mMaxNumResources, "Cannot register more than 10 resources.");

    // Get a handle
    int handle = mNextResourceHandle;
    mNextResourceHandle++;
    // Set the number of stages
    mNumStages[handle] = numStages;

    return handle;
  }

  _CCCL_API void constexpr registerPhase(int resourceHandle, int numOwningThreads, uint64_t* ptrBar)
  {
    _WS_CONSTANT_ASSERT(!mHasInitialized, "Cannot register phase after SyncHandler has been initialized.");
    _WS_CONSTANT_ASSERT(resourceHandle < mNextResourceHandle, "Invalid resource handle.");

    // Get phase index:
    int curPhase = mNumPhases[resourceHandle];
    _WS_CONSTANT_ASSERT(curPhase < mMaxNumPhases, "Cannot register more phases than maximum.");

    mNumOwningThreads[resourceHandle][curPhase] = numOwningThreads;
    mPtrBar[resourceHandle][curPhase]           = ptrBar;

    mNumPhases[resourceHandle]++;
  }

  // clusterInitSync can only be called on device.
  template <int NumThreads>
  _CCCL_DEVICE_API void clusterInitSync(SpecialRegisters sr, SkipSync)
  {
    _WS_CONSTANT_ASSERT(!mHasInitialized, "Cannot initialize SyncHandler twice.");
    mHasInitialized = true;

    // All warps iterate through all resources and phases. Since all array indices have to be statically resolved by the
    // SROA optimization to avoid spilling to local memory, we cannot split the iteration among warps etc.
    for (int ri = 0; ri < mMaxNumResources; ri++)
    {
      if (ri >= mNextResourceHandle)
      {
        break;
      }
      const int resNumPhases = mNumPhases[ri];
      const int numStages    = mNumStages[ri];

      for (int pi = 0; pi < mMaxNumPhases; pi++)
      {
        if (pi >= resNumPhases)
        {
          break;
        }

        uint64_t* ptrBar     = mPtrBar[ri][pi];
        int numOwningThreads = mNumOwningThreads[ri][pi];
        // use block strided iteration to vectorize setup of barriers
        for (int si = sr.threadIdxX; si < numStages; si += NumThreads)
        {
          ::cuda::ptx::mbarrier_init(&ptrBar[si], numOwningThreads);
        }
      }
    }
  }

  template <int NumThreads>
  _CCCL_DEVICE_API void clusterInitSync(SpecialRegisters sr)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   clusterInitSync<NumThreads>(sr, SkipSync{});
                   __cluster_barrier_arrive_relaxed();
                   __cluster_barrier_wait();
                 }))
  }
};
} // namespace detail::warpspeed

CUB_NAMESPACE_END
