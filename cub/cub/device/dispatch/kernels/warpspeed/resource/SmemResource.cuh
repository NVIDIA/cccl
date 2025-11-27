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

template <typename _Tp>
struct SmemResource : SmemResourceRaw
{
  _CCCL_DEVICE_API SmemResource(SmemResourceRaw smemResourceRaw) noexcept
      : SmemResourceRaw(smemResourceRaw)
  {}

  [[nodiscard]] _CCCL_DEVICE_API SmemStage<_Tp> popStage() noexcept
  {
    return SmemStage<_Tp>(*this);
  }
};

template <typename StageType, int stageCount>
[[nodiscard]] _CCCL_API inline SmemResource<StageType>
makeSmemResource(SyncHandler& syncHandler, StageType (&smemBuffer)[stageCount])
{
  int sizeBytes = sizeof(smemBuffer[0]);
  int stride    = sizeof(smemBuffer[0]);

  auto raw = SmemResourceRaw(syncHandler, smemBuffer, sizeBytes, stride, stageCount);
  return SmemResource<StageType>(raw);
}

template <typename StageType>
[[nodiscard]] _CCCL_API inline SmemResource<StageType>
makeSmemResource(SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages, Elems elems = Elems{1})
{
  int align       = alignof(StageType);
  int sizeBytes   = elems.value() * sizeof(StageType);
  int strideBytes = sizeBytes;

  void* ptrBase = smemAllocator.alloc(stages.value() * strideBytes, align);
  auto raw      = SmemResourceRaw(syncHandler, ptrBase, sizeBytes, strideBytes, stages.value());
  return SmemResource<StageType>(raw);
}

CUB_NAMESPACE_END
