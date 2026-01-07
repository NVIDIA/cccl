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

#include <cub/detail/warpspeed/makeWarpUniform.cuh> // makeWarpUniform.cuh

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/cstdint> // uint32_t

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
// Commonly used special registers that we should cache in registers or uniform
// registers.
struct SpecialRegisters
{
  const uint32_t clusterCtaRank;
  const uint32_t blockIdxX;
  const uint32_t threadIdxX;
  const uint32_t warpIdx;
  const uint32_t laneIdx;
};

[[nodiscard]] _CCCL_DEVICE_API inline SpecialRegisters getSpecialRegisters()
{
  uint32_t clusterCtaRank = ::cuda::ptx::get_sreg_cluster_ctarank();
  uint32_t threadIdxX     = threadIdx.x;
  uint32_t warpIdx        = makeWarpUniform(threadIdxX / 32);
  return {clusterCtaRank, blockIdx.x, threadIdxX, warpIdx, ::cuda::ptx::get_sreg_laneid()};
}
} // namespace detail::scan

CUB_NAMESPACE_END
