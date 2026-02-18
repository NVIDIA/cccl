//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H
#define _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_comparable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief `equal_to_value` is a function object that checks if a value is equal to a stored value.
//! @tparam _Tp The type of the value to be compared.
template <typename _Tp>
struct equal_to_value
{
  _Tp __value_;

  _CCCL_API explicit constexpr equal_to_value(const _Tp& __value) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Tp>)
      : __value_(__value)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(::cuda::std::__is_cpp17_equality_comparable_v<_Tp, _Up>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Up& __lhs) const
    noexcept(::cuda::std::__is_cpp17_nothrow_equality_comparable_v<_Tp, _Up>)
  {
    return static_cast<bool>(__lhs == __value_);
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H
