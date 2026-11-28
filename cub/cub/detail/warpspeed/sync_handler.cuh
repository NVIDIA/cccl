// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/constant_assert.cuh>
#include <cub/detail/warpspeed/special_registers.cuh>

#include <cuda/__ptx/instructions/mbarrier_init.h>
#include <cuda/__ptx/ptx_helper_functions.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
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

  struct phase_barriers
  {
    ::cuda::std::uint32_t smem_base;
    int first_barrier;
    int owning_threads;
  };

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::uint32_t barrier_byte_offset(int index)
  {
    return static_cast<::cuda::std::uint32_t>(index * sizeof(::cuda::std::uint64_t));
  }

  int mNextResourceHandle = 0;
  int mNumStages[mMaxNumResources]{};
  ::cuda::std::uint8_t mNumPhases[mMaxNumResources]{};

  // Each registered phase contributes one barrier per stage to the block's initialization work.
  static constexpr int max_phases = mMaxNumResources * mMaxNumPhases;
  phase_barriers mPhases[max_phases]{};
  int mPhaseCount   = 0;
  int mBarrierCount = 0;

  _CCCL_HIDE_FROM_ABI constexpr SyncHandler() = default;

  // we need constant destruction for the host side single stage SMEM amount, which is only possible in C++20
#if _CCCL_STD_VER >= 2020
  _CCCL_HOST_DEVICE_API constexpr ~SyncHandler()
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
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int registerResource(int numStages)
  {
    _WS_CONSTANT_ASSERT(!mHasInitialized, "Cannot register resource after SyncHandler has been initialized.");
    // Avoid exceeding the max number of stages
    _WS_CONSTANT_ASSERT(mNextResourceHandle < mMaxNumResources, "Cannot register more than 10 resources.");

    // Get a handle
    const int handle = mNextResourceHandle;
    mNextResourceHandle++;
    // Set the number of stages
    mNumStages[handle] = numStages;

    return handle;
  }

  _CCCL_HOST_DEVICE_API constexpr void
  registerPhase(int resourceHandle, int numOwningThreads, [[maybe_unused]] ::cuda::std::uint64_t* ptrBar)
  {
    _WS_CONSTANT_ASSERT(!mHasInitialized, "Cannot register phase after SyncHandler has been initialized.");
    _WS_CONSTANT_ASSERT(resourceHandle < mNextResourceHandle, "Invalid resource handle.");
    _WS_CONSTANT_ASSERT(mNumPhases[resourceHandle] < mMaxNumPhases, "Cannot register more phases than maximum.");

    // The host and constant-evaluation paths only compute the shared-memory footprint.
    ::cuda::std::uint32_t smem_base = 0;
    if (!::cuda::std::is_constant_evaluated())
    {
      // Bias the shared address by the logical index of this phase's first barrier. Unsigned byte offsets also
      // work when the phases occupy shared memory in a different order from registration.
      NV_IF_TARGET(NV_IS_DEVICE, (smem_base = ::cuda::ptx::__as_ptr_smem(ptrBar) - barrier_byte_offset(mBarrierCount);))
    }
    mPhases[mPhaseCount++] = {smem_base, mBarrierCount, numOwningThreads};
    mBarrierCount += mNumStages[resourceHandle];
    mNumPhases[resourceHandle]++;
  }

  // clusterInitSync can only be called on device.
  template <int NumThreads>
  _CCCL_DEVICE_API void clusterInitSync(SpecialRegisters sr, SkipSync)
  {
    _WS_CONSTANT_ASSERT(!mHasInitialized, "Cannot initialize SyncHandler twice.");
    mHasInitialized = true;

    // Neighboring lanes initialize neighboring logical barriers, including those in different phases.
    for (int index = static_cast<int>(sr.threadIdxX); index < mBarrierCount; index += NumThreads)
    {
      // The last phase starting at or before index owns this barrier.
      phase_barriers phase = mPhases[0];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < mPhaseCount; ++i)
      {
        if (index >= mPhases[i].first_barrier)
        {
          phase = mPhases[i];
        }
      }
      auto* barrier = ::cuda::ptx::__from_ptr_smem<::cuda::std::uint64_t>(phase.smem_base + barrier_byte_offset(index));
      ::cuda::ptx::mbarrier_init(barrier, phase.owning_threads);
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
