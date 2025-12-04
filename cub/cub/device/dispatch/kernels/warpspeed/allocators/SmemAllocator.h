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

#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
struct SmemAllocator
{
  uint32_t mPtrSmem32 = 0;
  int mAllocatedSize  = 0;

  _CCCL_API constexpr SmemAllocator()
  {
    // we only need the real pointer at runtime in device code
    if (!::cuda::std::is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, mPtrSmem32 = dynamic_smem_base();)
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static uint32_t dynamic_smem_base() noexcept
  {
    extern __shared__ char warpSpeedDynamicSmemBase[];
    return __cvta_generic_to_shared(warpSpeedDynamicSmemBase);
  }

  // SmemAllocator is a non-copyable, non-movable type. It must be passed by
  // (mutable) reference to be useful.
  SmemAllocator(const SmemAllocator&)             = delete; // Delete copy constructor
  SmemAllocator(SmemAllocator&&)                  = delete; // Delete move constructor
  SmemAllocator& operator=(const SmemAllocator&)  = delete; // Delete copy assignment
  SmemAllocator& operator=(const SmemAllocator&&) = delete; // Delete move assignment

  [[nodiscard]] _CCCL_API constexpr void* alloc(uint32_t size, uint32_t align = 0)
  {
    // Align mPtrSmem32 to requested alignment (round-up)
    uint32_t ptrAllocation32 = (mPtrSmem32 + (align - 1)) & ~(align - 1);

    // Move base pointer and update allocated size
    mAllocatedSize += size + ptrAllocation32 - mPtrSmem32;
    mPtrSmem32 = ptrAllocation32 + size;

    // we only need the pointer at runtime in device code
    if (!::cuda::std::is_constant_evaluated())
    {
      NV_IF_TARGET(
        NV_IS_DEVICE,
        (
          // Convert allocated smem address to generic pointer
          void* mPtrAllocation = __cvta_shared_to_generic(ptrAllocation32);
          // Ensure alignment calculation does not move down into rest of kernel code.
          return optimizeSmemPtr(mPtrAllocation);))
    }
    return nullptr;
  }

  [[nodiscard]] _CCCL_API constexpr uint32_t sizeBytes() const
  {
    return mAllocatedSize;
  }
};
} // namespace detail::scan

CUB_NAMESPACE_END
