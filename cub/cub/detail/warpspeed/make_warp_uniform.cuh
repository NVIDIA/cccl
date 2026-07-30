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

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
// Move register to uniform register

// For int32_t and uint32_t, we can use the CREDUX instruction, which is coupled and has a constant latency.
// For 64-bit types, we still use __shfl_sync

[[nodiscard]] _CCCL_DEVICE_API inline int makeWarpUniform(int x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __reduce_min_sync(~0, x);), (return x;));
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t makeWarpUniform(::cuda::std::uint32_t x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __reduce_min_sync(~0, x);), (return x;));
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint64_t makeWarpUniform(::cuda::std::uint64_t x)
{
  return __shfl_sync(~0, x, 0);
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
