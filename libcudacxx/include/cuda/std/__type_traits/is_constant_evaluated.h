//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H

#include <cuda/std/detail/__config>

#include "cuda/std/detail/libcxx/include/__config"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_CONSTANT_EVALUATED)
_LIBCUDACXX_HIDE_FROM_ABI inline constexpr bool is_constant_evaluated() noexcept
{
  return _LIBCUDACXX_IS_CONSTANT_EVALUATED();
}

inline constexpr _LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_is_constant_evaluated() noexcept
{
  return _LIBCUDACXX_IS_CONSTANT_EVALUATED();
}
inline constexpr _LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_default_is_constant_evaluated() noexcept
{
  return _LIBCUDACXX_IS_CONSTANT_EVALUATED();
}
#else // ^^^ _LIBCUDACXX_IS_CONSTANT_EVALUATED ^^^ / vvv !_LIBCUDACXX_IS_CONSTANT_EVALUATED vvv
inline constexpr _LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_is_constant_evaluated() noexcept
{
  return false;
}
inline constexpr _LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_default_is_constant_evaluated() noexcept
{
  return true;
}
#endif // !_LIBCUDACXX_IS_CONSTANT_EVALUATED

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CONSTANT_EVALUATED_H
