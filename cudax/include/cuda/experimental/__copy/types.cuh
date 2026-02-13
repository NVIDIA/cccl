//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_TYPES_H
#define __CUDAX_COPY_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/array>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/// @brief Raw tensor with dynamic shapes and strides
template <typename _Tp, ::cuda::std::size_t _MaxRank>
struct __raw_tensor
{
  _Tp* __data;
  ::cuda::std::size_t __rank;
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __shapes;
  ::cuda::std::array<::cuda::std::int64_t, _MaxRank> __strides;
};

/// @brief Raw tensor with dynamic shapes and strides, ordered by stride in descending order
template <typename _Tp, ::cuda::std::size_t _MaxRank>
struct __raw_tensor_ordered : __raw_tensor<_Tp, _MaxRank>
{
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __orders;
};

enum class __copy_direction
{
  __host_to_device,
  __device_to_host,
};
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_TYPES_H
