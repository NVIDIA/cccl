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

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Raw tensor descriptor with dynamic rank, extents, and strides.
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
struct __raw_tensor
{
  using __rank_t = ::cuda::std::size_t;

  _Tp* __data;
  __rank_t __rank;
  ::cuda::std::array<_ExtentT, _MaxRank> __extents;
  ::cuda::std::array<_StrideT, _MaxRank> __strides;
};

//! @brief Direction of an asynchronous memcpy operation.
enum class __copy_direction
{
  __host_to_device,
  __device_to_host,
};
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_TYPES_H
