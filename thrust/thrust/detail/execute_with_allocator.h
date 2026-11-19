// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/execute_with_allocator_fwd.h>
#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename Allocator, typename T, typename = void>
inline constexpr bool has_rebind_other_v = false;

template <typename Allocator, typename T>
inline constexpr bool
  has_rebind_other_v<Allocator, T, ::cuda::std::void_t<typename Allocator::template rebind<T>::other>> = true;

// Like allocator_traits::rebind_alloc, but false instead of a hard error. Without a nested rebind, only
// allocators whose first template argument is their value_type are rebound (not e.g. alloc<Pool>).
template <typename Allocator, typename T>
inline constexpr bool is_rebindable_allocator_v = has_rebind_other_v<Allocator, T>;

template <template <typename, typename...> class Allocator, typename U, typename... Args, typename T>
inline constexpr bool is_rebindable_allocator_v<Allocator<U, Args...>, T> =
  has_rebind_other_v<Allocator<U, Args...>, T> || ::cuda::std::is_same_v<U, typename Allocator<U, Args...>::value_type>;

template <typename Allocator, typename T, bool = is_rebindable_allocator_v<Allocator, T>>
struct temporary_buffer_allocator_impl
{
  using type = Allocator;
};

template <typename Allocator, typename T>
struct temporary_buffer_allocator_impl<Allocator, T, true>
{
  using rebound = typename ::cuda::std::allocator_traits<Allocator>::template rebind_alloc<T>;
  using type    = ::cuda::std::conditional_t<::cuda::std::is_constructible_v<rebound, Allocator&>, rebound, Allocator>;
};

// Allocators that cannot be rebound (e.g. non-template allocators of char) keep being used as they are.
template <typename Allocator, typename T>
using temporary_buffer_allocator_t = typename temporary_buffer_allocator_impl<Allocator, T>::type;

// decltype(auto): returns either a reference to alloc or a rebound copy
template <typename T, typename Allocator>
_CCCL_HOST_API decltype(auto) temporary_buffer_allocator(Allocator& alloc)
{
  if constexpr (::cuda::std::is_same_v<temporary_buffer_allocator_t<Allocator, T>, Allocator>)
  {
    static_assert(!::cuda::std::is_void_v<typename ::cuda::std::allocator_traits<Allocator>::value_type>,
                  "an allocator with value_type void must support rebinding");
    return (alloc);
  }
  else
  {
    return temporary_buffer_allocator_t<Allocator, T>(alloc);
  }
}

template <typename T, typename Allocator, template <typename> class BaseSystem>
_CCCL_HOST_API ::cuda::std::pair<T*, std::ptrdiff_t>
get_temporary_buffer(thrust::detail::execute_with_allocator<Allocator, BaseSystem>& system, std::ptrdiff_t n)
{
  using naked_allocator = temporary_buffer_allocator_t<::cuda::std::remove_reference_t<Allocator>, T>;
  using alloc_traits    = ::cuda::std::allocator_traits<naked_allocator>;
  using void_pointer    = typename alloc_traits::void_pointer;
  using size_type       = typename alloc_traits::size_type;
  using value_type      = typename alloc_traits::value_type;

  // How many elements of type value_type do we need to accommodate n elements
  // of type T?
  const size_type num_elements = static_cast<size_type>(::cuda::ceil_div(sizeof(T) * n, sizeof(value_type)));

  auto&& alloc           = temporary_buffer_allocator<T>(system.get_allocator());
  const void_pointer ptr = alloc_traits::allocate(alloc, num_elements);

  // Return the pointer and the number of elements of type T allocated.
  return ::cuda::std::make_pair(thrust::reinterpret_pointer_cast<T*>(ptr), n);
}

template <typename Pointer, typename Allocator, template <typename> class BaseSystem>
_CCCL_HOST_API void return_temporary_buffer(
  thrust::detail::execute_with_allocator<Allocator, BaseSystem>& system, Pointer p, std::ptrdiff_t n)
{
  using T               = typename ::cuda::std::pointer_traits<Pointer>::element_type;
  using naked_allocator = temporary_buffer_allocator_t<::cuda::std::remove_reference_t<Allocator>, T>;
  using alloc_traits    = ::cuda::std::allocator_traits<naked_allocator>;
  using pointer         = typename alloc_traits::pointer;
  using size_type       = typename alloc_traits::size_type;
  using value_type      = typename alloc_traits::value_type;

  const size_type num_elements = ::cuda::ceil_div(sizeof(T) * n, sizeof(value_type));

  const pointer to_ptr = thrust::reinterpret_pointer_cast<pointer>(p);
  auto&& alloc         = temporary_buffer_allocator<T>(system.get_allocator());
  alloc_traits::deallocate(alloc, to_ptr, num_elements);
}
} // namespace detail

THRUST_NAMESPACE_END
