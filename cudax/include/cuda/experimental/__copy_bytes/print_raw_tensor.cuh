//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_PRINT_RAW_TENSOR_H
#define __CUDAX_COPY_PRINT_RAW_TENSOR_H

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

#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cstdio>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Prints a raw tensor's extents and strides to stdout in the format `(extents):(strides)`.
//!
//! @param[in] __tensor Raw tensor to print
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __println(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor)
{
  const auto __rank = static_cast<int>(__tensor.__rank);
  ::printf("(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    ::printf("%llu, ", static_cast<unsigned long long>(__tensor.__extents[__i]));
  }
  if (__rank > 0)
  {
    ::printf("%llu", static_cast<unsigned long long>(__tensor.__extents[__rank - 1]));
  }
  ::printf("):(");
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    ::printf("%lld, ", static_cast<long long>(__tensor.__strides[__i]));
  }
  if (__rank > 0)
  {
    ::printf("%lld", static_cast<long long>(__tensor.__strides[__rank - 1]));
  }
  ::printf(")\n");
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_PRINT_RAW_TENSOR_H
