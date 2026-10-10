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

#include <cub/detail/warpspeed/allocators/smem_allocator.cuh>
#include <cub/detail/warpspeed/resource/smem_resource_raw.cuh>
#include <cub/detail/warpspeed/resource/smem_stage.cuh>
#include <cub/detail/warpspeed/sync_handler.cuh>
#include <cub/detail/warpspeed/values.cuh>

#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
template <typename Tp>
struct SmemResource : SmemResourceRaw
{
  template <int StageCount>
  _CCCL_HOST_DEVICE_API SmemResource(SyncHandler& syncHandler, Tp (&smemBuffer)[StageCount])
      : SmemResourceRaw(syncHandler, smemBuffer, sizeof(smemBuffer[0]), sizeof(smemBuffer[0]), StageCount)
  {}

  _CCCL_HOST_DEVICE_API constexpr SmemResource(
    SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages, Elems elems = Elems{1})
      : SmemResourceRaw(makeSmemResourceRaw(syncHandler, smemAllocator, stages, elems))
  {}

  [[nodiscard]] _CCCL_DEVICE_API SmemStage<Tp> nextStage() noexcept
  {
    return SmemStage<Tp>(*this);
  }

private:
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr SmemResourceRaw
  makeSmemResourceRaw(SyncHandler& syncHandler, SmemAllocator& smemAllocator, Stages stages, Elems elems = Elems{1})
  {
    const int align       = alignof(Tp);
    const int sizeBytes   = ::cuda::std::to_underlying(elems) * sizeof(Tp);
    const int strideBytes = sizeBytes;

    void* ptrBase = smemAllocator.alloc(::cuda::std::to_underlying(stages) * strideBytes, align);
    return {syncHandler, ptrBase, sizeBytes, strideBytes, ::cuda::std::to_underlying(stages)};
  }
};
} // namespace detail::warpspeed

CUB_NAMESPACE_END
