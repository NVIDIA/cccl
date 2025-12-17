/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/for_each.h>
#include <thrust/uninitialized_fill.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename Allocator>
struct construct1_via_allocator
{
  Allocator& a;

  template <typename T>
  inline _CCCL_HOST_DEVICE void operator()(T& x)
  {
    allocator_traits<Allocator>::construct(a, &x);
  }
};

// we need to construct T via the allocator if the Allocator does something interesting or if T's default constructor
// does something interesting
template <typename Allocator, typename T>
inline constexpr bool needs_default_construct_via_allocator =
  allocator_traits_detail::has_member_construct1<Allocator, T>::value
  || !::cuda::std::is_trivially_default_constructible_v<T>;

// we know that std::allocator::construct's only effect is to call T's
// default constructor, so we needn't use it for default construction
// unless T's constructor does something interesting
template <typename U, typename T>
inline constexpr bool needs_default_construct_via_allocator<std::allocator<U>, T> =
  !::cuda::std::is_trivially_default_constructible_v<T>;

template <typename Allocator, typename Pointer, typename Size>
_CCCL_HOST_DEVICE void value_initialize_range(Allocator& a, Pointer p, Size n)
{
  if constexpr (needs_default_construct_via_allocator<Allocator, typename pointer_element<Pointer>::type>)
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a), p, n, construct1_via_allocator<Allocator>{a});
  }
  else
  {
    thrust::uninitialized_fill_n(allocator_system<Allocator>::get(a), p, n, typename pointer_element<Pointer>::type());
  }
}
} // namespace detail
THRUST_NAMESPACE_END
