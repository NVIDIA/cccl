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

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
// Move register to uniform register

// For int32_t and uint32_t, we can use the new CREDUX instruction, which is
// coupled and has a constant 13 cycle latency.
// For 64-bit types, we still use __shfl_sync
[[nodiscard]] _CCCL_DEVICE_API inline int makeWarpUniform(int x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __reduce_min_sync(~0, x);), (return x;));
}
[[nodiscard]] _CCCL_DEVICE_API inline uint32_t makeWarpUniform(uint32_t x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return __reduce_min_sync(~0, x);), (return x;));
}
[[nodiscard]] _CCCL_DEVICE_API inline uint64_t makeWarpUniform(uint64_t x)
{
  return __shfl_sync(~0, x, 0);
}
} // namespace detail

CUB_NAMESPACE_END
