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

#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename... Args>
struct emplace_via_allocator
{
  ::cuda::std::tuple<Args...> args_as_a_tuple;

  template <typename T>
  _CCCL_HOST_DEVICE void operator()(T& loc)
  {
    ::cuda::std::apply(
      [&](auto&... xs) {
        new (static_cast<void*>(&loc)) T(xs...);
      },
      args_as_a_tuple);
  }
};

// Build one object at p from args, on the system (CPU or GPU) the allocator belongs to
template <typename Allocator, typename Pointer, typename... Args>
_CCCL_HOST_DEVICE void emplace_construct(Allocator& a, Pointer loc, Args... args)
{
  // Dispatch according to backend (loop/kernel)
  thrust::for_each_n(allocator_system<Allocator>::get(a), loc, 1, emplace_via_allocator<Args...>{{args...}});
}
} // namespace detail
THRUST_NAMESPACE_END
