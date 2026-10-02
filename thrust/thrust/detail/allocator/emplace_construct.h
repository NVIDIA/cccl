// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/allocator/allocator_system.h>
#include <thrust/for_each.h>

#include <cuda/std/__host_stdlib/memory>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/forward_like.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// emplace_construct has 2 cases:
// if Allocator has an effectful member function construct(T*, Args...):
//   1. construct via the allocator
// else
//   2. construct via placement new
// Both are dispatched with for_each_n on the allocator's system.

template <typename Allocator, typename T, typename... Args>
inline constexpr bool has_effectful_member_construct = ::cuda::std::__has_construct<Allocator, T*, Args...>;

// std::allocator::construct's only effect is to invoke placement new
template <typename U, typename T, typename... Args>
inline constexpr bool has_effectful_member_construct<std::allocator<U>, T, Args...> = false;

template <typename Allocator, typename... Args>
struct emplace_via_allocator_construct
{
  Allocator& a;
  // Arguments are stored as value here (decay) but forwarded like Args to ctor (forward_like)
  ::cuda::std::tuple<::cuda::std::decay_t<Args>...> args;

  template <typename T>
  _CCCL_HOST_DEVICE void operator()(T& loc)
  {
    ::cuda::std::apply(
      [&](auto&... xs) {
        ::cuda::std::allocator_traits<Allocator>::construct(a, &loc, ::cuda::std::forward_like<Args>(xs)...);
      },
      args);
  }
};

template <typename... Args>
struct emplace_via_placement_new
{
  ::cuda::std::tuple<::cuda::std::decay_t<Args>...> args;

  template <typename T>
  _CCCL_HOST_DEVICE void operator()(T& loc)
  {
    ::cuda::std::apply(
      [&](auto&... xs) {
        ::new (static_cast<void*>(&loc)) T(::cuda::std::forward_like<Args>(xs)...);
      },
      args);
  }
};

// Build one object at loc from args, on the system (CPU or GPU) the allocator belongs to
template <typename Allocator, typename Pointer, typename... Args>
_CCCL_HOST_DEVICE void emplace_construct(Allocator& a, Pointer loc, Args&&... args)
{
  using T = typename ::cuda::std::pointer_traits<Pointer>::element_type;
  if constexpr (has_effectful_member_construct<Allocator, T, Args...>)
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a),
                       loc,
                       1,
                       emplace_via_allocator_construct<Allocator, Args...>{a, {::cuda::std::forward<Args>(args)...}});
  }
  else
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a),
                       loc,
                       1,
                       emplace_via_placement_new<Args...>{{::cuda::std::forward<Args>(args)...}});
  }
}
} // namespace detail
THRUST_NAMESPACE_END
