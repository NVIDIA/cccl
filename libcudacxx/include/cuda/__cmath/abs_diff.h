//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_ABS_DIFF_H
#define _CUDA___CMATH_ABS_DIFF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::make_unsigned_t<_Tp> __abs_diff_impl_generic(_Tp __x, _Tp __y) noexcept
{
  using _Up _CCCL_NODEBUG = ::cuda::std::make_unsigned_t<_Tp>;

  const auto __minuend    = static_cast<_Up>((__x > __y) ? __x : __y);
  const auto __subtrahend = static_cast<_Up>((__x > __y) ? __y : __x);
  return static_cast<_Up>(__minuend - __subtrahend);
}

#if _CCCL_CUDA_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::make_unsigned_t<_Tp> __abs_diff_impl_device(_Tp __x, _Tp __y) noexcept
{
  using _Up _CCCL_NODEBUG = ::cuda::std::make_unsigned_t<_Tp>;

  if constexpr (::cuda::std::is_signed_v<_Tp> && sizeof(_Tp) <= sizeof(::cuda::std::int32_t))
  {
    return static_cast<_Up>(::__sad(__x, __y, 0));
  }
  else if constexpr (::cuda::std::is_unsigned_v<_Tp> && sizeof(_Tp) <= sizeof(::cuda::std::uint32_t))
  {
    return static_cast<_Up>(::__usad(__x, __y, 0));
  }
  else
  {
    return ::cuda::__abs_diff_impl_generic(__x, __y);
  }
}
#endif // _CCCL_CUDA_COMPILATION()

//! @brief Computes absolute difference.
//! @param[in] __lhs The left-hand side input.
//! @param[in] __rhs The right-hand side input.
//! @return An unsigned value containing the absolute difference of each pair of input elements.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr ::cuda::std::make_unsigned_t<_Tp> abs_diff(_Tp __lhs, _Tp __rhs) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_IS_DEVICE, ({ return ::cuda::__abs_diff_impl_device(__lhs, __rhs); }))
  }
  return ::cuda::__abs_diff_impl_generic(__lhs, __rhs);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_ABS_DIFF_H
