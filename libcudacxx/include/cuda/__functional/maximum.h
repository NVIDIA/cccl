//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_FUNCTIONAL_MAXIMUM_H
#define _CUDA_FUNCTIONAL_MAXIMUM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/minimum_maximum_common.h>
#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__utility/ctad_support.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT maximum
{
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __lhs, const _Tp& __rhs) const
    noexcept(__is_maximum_minimum_noexcept_v<_Tp, _Tp, _Tp>)
  {
    if constexpr (::cuda::std::is_floating_point_v<_Tp> || ::cuda::std::__is_extended_floating_point_v<_Tp>)
    {
      return ::cuda::std::fmax(__lhs, __rhs);
    }
    else
    {
      return (__lhs < __rhs) ? __rhs : __lhs;
    }
  }
};
_CCCL_CTAD_SUPPORTED_FOR_TYPE(maximum);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT maximum<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class Tp, class Up, class _Common = ::cuda::std::common_type_t<Tp, Up>>
  [[nodiscard]] _CCCL_API constexpr _Common operator()(const Tp& __lhs, const Up& __rhs) const
    noexcept(__is_maximum_minimum_noexcept_v<Tp, Up, _Common>)
  {
    if constexpr (::cuda::std::is_floating_point_v<_Common> || ::cuda::std::__is_extended_floating_point_v<_Common>)
    {
      return ::cuda::std::fmax(static_cast<_Common>(__lhs), static_cast<_Common>(__rhs));
    }
    else
    {
      return (__lhs < __rhs) ? __rhs : __lhs;
    }
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_MAXIMUM_H
