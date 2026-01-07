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

#include <cub/detail/warpspeed/allocators/SmemAllocator.h>
#include <cub/detail/warpspeed/resource/SmemResourceRaw.cuh>
#include <cub/detail/warpspeed/resource/SmemStage.cuh>
#include <cub/detail/warpspeed/SyncHandler.h>
#include <cub/detail/warpspeed/values.h>

#include <cuda/std/__utility/to_underlying.h>
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

  _CCCL_API constexpr SmemResource(
    SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages, Elems elems = Elems{1})
      : SmemResourceRaw(makeSmemResourceRaw(syncHandler, smemAllocator, stages, elems))
  {}

  [[nodiscard]] _CCCL_DEVICE_API SmemStage<_Tp> popStage() noexcept
  {
    return SmemStage<_Tp>(*this);
  }

private:
  [[nodiscard]] _CCCL_API static constexpr inline SmemResourceRaw
  makeSmemResourceRaw(SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages, Elems elems = Elems{1})
  {
    int align       = alignof(_Tp);
    int sizeBytes   = ::cuda::std::to_underlying(elems) * sizeof(_Tp);
    int strideBytes = sizeBytes;

    void* ptrBase = smemAllocator.alloc(::cuda::std::to_underlying(stages) * strideBytes, align);
    return SmemResourceRaw(syncHandler, ptrBase, sizeBytes, strideBytes, ::cuda::std::to_underlying(stages));
  }
};
} // namespace detail::scan

CUB_NAMESPACE_END
