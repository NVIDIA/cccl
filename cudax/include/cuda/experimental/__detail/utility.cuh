//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_UTILITY_H
#define __CUDAX_DETAIL_UTILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/declval.h>

namespace cuda::experimental
{
namespace detail
{
// This is a helper type that can be used to ignore function arguments.
struct [[maybe_unused]] __ignore
{
  _CCCL_HIDE_FROM_ABI __ignore() = default;

  template <typename... _Args>
  _CCCL_HOST_DEVICE constexpr __ignore(_Args&&...) noexcept
  {}
};

// Classes can inherit from this type to become immovable.
struct __immovable
{
  _CCCL_HIDE_FROM_ABI __immovable()              = default;
  __immovable(__immovable&&) noexcept            = delete;
  __immovable& operator=(__immovable&&) noexcept = delete;
};

template <class... _Types>
struct _CCCL_DECLSPEC_EMPTY_BASES __inherit : _Types...
{};

template <class _Type, template <class...> class _Template>
inline constexpr bool __is_specialization_of = false;

template <template <class...> class _Template, class... _Args>
inline constexpr bool __is_specialization_of<_Template<_Args...>, _Template> = true;

} // namespace detail

template <class _Tp>
using __identity_t _CCCL_NODEBUG_ALIAS = _Tp;

using _CUDA_VSTD::declval;

struct uninit_t
{
  explicit uninit_t() = default;
};

_CCCL_GLOBAL_CONSTANT uninit_t uninit{};
} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_UTILITY_H
