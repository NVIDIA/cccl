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
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemResourceRaw.cuh>
#include <cub/device/dispatch/kernels/warpspeed/resource/SmemStage.cuh>
#include <cub/device/dispatch/kernels/warpspeed/SyncHandler.h>
#include <cub/device/dispatch/kernels/warpspeed/values.h>

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
template <typename _Tp>
struct SmemResource : SmemResourceRaw
{
  template <int stageCount>
  _CCCL_API SmemResource(SyncHandler& syncHandler, _Tp (&smemBuffer)[stageCount])
      : SmemResourceRaw(syncHandler, smemBuffer, sizeof(smemBuffer[0]), sizeof(smemBuffer[0]), stageCount)
  {}

  _CCCL_API SmemResource(SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages, Elems elems = Elems{1})
      : SmemResourceRaw(makeSmemResourceRaw(syncHandler, smemAllocator, stages, elems))
  {}

  [[nodiscard]] _CCCL_DEVICE_API SmemStage<_Tp> popStage() noexcept
  {
    return SmemStage<_Tp>(*this);
  }

private:
  [[nodiscard]] _CCCL_API static inline SmemResourceRaw
  makeSmemResourceRaw(SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages, Elems elems = Elems{1})
  {
    int align       = alignof(_Tp);
    int sizeBytes   = elems.value() * sizeof(_Tp);
    int strideBytes = sizeBytes;

    void* ptrBase = smemAllocator.alloc(stages.value() * strideBytes, align);
    return SmemResourceRaw(syncHandler, ptrBase, sizeBytes, strideBytes, stages.value());
  }
};
} // namespace detail::scan

CUB_NAMESPACE_END
