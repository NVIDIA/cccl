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

#include <cub/device/dispatch/kernels/warpspeed/optimizeSmemPtr.cuh>

#include <cuda/std/cstdint>

#include <nv/target>

struct SmemAllocator
{
  uint32_t mPtrSmem32 = 0;
  int mAllocatedSize  = 0;

  _CCCL_API SmemAllocator()
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (extern __shared__ char warpSpeedDynamicSmemBase[];
                  mPtrSmem32 = __cvta_generic_to_shared(warpSpeedDynamicSmemBase);))
  }

  // SmemAllocator is a non-copyable, non-movable type. It must be passed by
  // (mutable) reference to be useful.
  SmemAllocator(const SmemAllocator&)             = delete; // Delete copy constructor
  SmemAllocator(SmemAllocator&&)                  = delete; // Delete move constructor
  SmemAllocator& operator=(const SmemAllocator&)  = delete; // Delete copy assignment
  SmemAllocator& operator=(const SmemAllocator&&) = delete; // Delete move assignment

  [[nodiscard]] _CCCL_API void* alloc(uint32_t size, uint32_t align = 0)
  {
    // Align mPtrSmem32 to requested alignment (round-up)
    uint32_t ptrAllocation32 = (mPtrSmem32 + (align - 1)) & ~(align - 1);

    // Move base pointer and update allocated size
    mAllocatedSize += size + ptrAllocation32 - mPtrSmem32;
    mPtrSmem32 = ptrAllocation32 + size;

    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (
        // Convert allocated smem address to generic pointer
        void* mPtrAllocation = __cvta_shared_to_generic(ptrAllocation32);
        // Ensure alignment calculation does not move down into rest of kernel code.
        return optimizeSmemPtr(mPtrAllocation);),
      (return nullptr;))
  }

  [[nodiscard]] _CCCL_API uint32_t sizeBytes() const
  {
    return mAllocatedSize;
  }
};
