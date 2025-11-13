//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_THREAD_LEVEL_H
#define _CUDA___HIERARCHY_THREAD_LEVEL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/hierarchy.h>
#include <cuda/__hierarchy/hierarchy_query_result.h>
#include <cuda/__hierarchy/native_hierarchy_level_base.h>
#include <cuda/std/__mdspan/extents.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct thread_level : __native_hierarchy_level_base<thread_level>
{
  using __next_level = block_level;

  using __base_type = __native_hierarchy_level_base<thread_level>;
  using __base_type::extents;
  using __base_type::index;

  // interactions with block level

  [[nodiscard]] _CCCL_DEVICE_API static ::cuda::std::dims<3, unsigned> extents(const block_level&) noexcept
  {
    return ::cuda::std::dims<3, unsigned>{blockDim.z, blockDim.y, blockDim.x};
  }
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<unsigned, 3> index(const block_level&) noexcept
  {
    return {threadIdx.z, threadIdx.y, threadIdx.x};
  }

  // interactions with warp level

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::extents<unsigned, 32> extents(const warp_level&) noexcept
  {
    return {};
  }
  [[nodiscard]] _CCCL_DEVICE_API static hierarchy_query_result<unsigned, 1> index(const warp_level&) noexcept
  {
    return {::cuda::ptx::get_sreg_laneid()};
  }
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

_CCCL_GLOBAL_CONSTANT thread_level thread;

_CCCL_END_NAMESPACE_CUDA_DEVICE

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___HIERARCHY_THREAD_LEVEL_H
