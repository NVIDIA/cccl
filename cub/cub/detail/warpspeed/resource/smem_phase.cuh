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

#include <cub/detail/warpspeed/resource/smem_ref.cuh>
#include <cub/detail/warpspeed/resource/smem_resource_raw.cuh>

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
template <typename _Tp>
struct SmemPhase
{
  SmemResourceRaw& mSmemResourceRaw;
  int mCurPhase;

  _CCCL_DEVICE_API SmemPhase(SmemResourceRaw& smemResourceRaw, int phase) noexcept
      : mSmemResourceRaw(smemResourceRaw)
      , mCurPhase(phase)
  {}

  [[nodiscard]] _CCCL_DEVICE_API SmemRef<_Tp> acquireRef()
  {
    // Wait on barrier
    mSmemResourceRaw.acquire(mCurPhase);
    // Return ref
    return SmemRef<_Tp>(mSmemResourceRaw, mCurPhase);
  }
};
} // namespace detail::scan

CUB_NAMESPACE_END
