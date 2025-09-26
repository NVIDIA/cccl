//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_IN_RANGE_H
#define _CUDA___UTILITY_IN_RANGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/cmp.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(typename _Tp, typename _Up)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Up>)
[[nodiscard]] _CCCL_API constexpr bool in_range(_Tp __v, _Up __start, _Up __end) noexcept
{
  _CCCL_ASSERT(__end > __start, "in_range: __end must be greater than __start");
  if constexpr (::cuda::std::is_unsigned_v<_Tp>)
  {
    // if __end > __start, we know that the range is always positive. Similarly, __v is positive if unsigned.
    // this optimization is useful when __start and __end are compile-time constants, or when in_range is used multiple
    // times with the same range
    using _CommonType         = ::cuda::std::common_type_t<_Tp, _Up, unsigned>; // at least 32-bit
    using _UnsignedCommonType = ::cuda::std::make_unsigned_t<_CommonType>;
    const auto __start1       = static_cast<_UnsignedCommonType>(__start);
    const auto __end1         = static_cast<_UnsignedCommonType>(__end);
    const auto __v1           = static_cast<_UnsignedCommonType>(__v);
    const auto __range        = __end1 - __start1;
    return (__v1 - __start1) < __range;
  }
  else
  {
    return ::cuda::std::cmp_greater_equal(__v, __start) && ::cuda::std::cmp_less(__v, __end);
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_IN_RANGE_H
