// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/functional.h>

#include <cuda/std/type_traits>

// TODO: move to libcu++

THRUST_NAMESPACE_BEGIN

namespace detail
{
// need a separate implementation trait because we SFINAE with a type parameter before the variadic pack
template <typename F, typename SFINAE, typename... Args>
struct can_copy_arguments_impl : std::false_type
{};

template <typename F, typename... Args>
struct can_copy_arguments_impl<F, ::cuda::std::void_t<decltype(F::can_copy_arguments)>, Args...>
{
  static constexpr bool value = F::can_copy_arguments;
};
} // namespace detail

// TODO(bgruber): bikeshed name
/// Trait telling whether a function object relies on the memory address of its arguments when called with the given set
/// of types. The nested value is true when the addresses of the arguments do not matter and arguments can be provided
/// from arbitrary copies of the respective sources.
template <typename F, typename... Args>
using can_copy_arguments = detail::can_copy_arguments_impl<F, void, Args...>;
// georgii: can_copy_arguments, maybe for review meeting

template <typename F>
struct copied_arguments_allowing_wrapper : F
{
  using F::operator();
  static constexpr bool can_copy_arguments = true;
};

// TODO(bgruber): bikeshed name, also consider "proclaim_something"
/// Creates a new function object from an existing one, allowing its arguments to be copies of whatever source they come
/// from. This implies that the addresses of the arguments are irrelevant to the function object.
template <typename F>
_CCCL_HOST_DEVICE constexpr auto allow_copied_arguments(F f) -> copied_arguments_allowing_wrapper<F>
{
  return copied_arguments_allowing_wrapper<F>{std::move(f)};
}

THRUST_NAMESPACE_END
