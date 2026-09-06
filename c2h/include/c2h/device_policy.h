// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/detail/__config>

#include <thrust/execution_policy.h>
#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/stream>
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#include <c2h/checked_allocator.cuh>

namespace c2h
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
// These policies are constructed from a checked allocator whose (non-noexcept) construction clang-tidy conservatively
// treats as potentially throwing during static initialization.
// NOLINTBEGIN(bugprone-throwing-static-initialization)
static const auto device_policy        = THRUST_NS_QUALIFIER::cuda::par(checked_cuda_allocator<char>{});
static const auto nosync_device_policy = THRUST_NS_QUALIFIER::cuda::par_nosync(checked_cuda_allocator<char>{});
// NOLINTEND(bugprone-throwing-static-initialization)
#else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
static const auto device_policy        = THRUST_NS_QUALIFIER::device;
static const auto nosync_device_policy = THRUST_NS_QUALIFIER::device;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
[[nodiscard]] inline auto device_policy_on(::cuda::stream_ref stream)
{
#  if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  return device_policy.on(stream.get());
#  else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  (void) stream;
  return device_policy;
#  endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
}
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
} // namespace c2h
