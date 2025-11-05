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

#include <cuda/std/__cccl/memory_wrapper.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// fill_construct_range has 2 cases:
// if Allocator has an effectful member function construct:
//   1. construct via the allocator
// else
//   2. construct via uninitialized_fill

template <typename Allocator, typename T, typename Arg1>
inline constexpr bool has_effectful_member_construct2 =
  allocator_traits_detail::has_member_construct2<Allocator, T, Arg1>::value;

// std::allocator::construct's only effect is to invoke placement new
template <typename U, typename T, typename Arg1>
inline constexpr bool has_effectful_member_construct2<std::allocator<U>, T, Arg1> = false;

template <typename Allocator, typename Arg1>
struct construct2_via_allocator
{
  Allocator& a;
  Arg1 arg;

  template <typename T>
  inline _CCCL_HOST_DEVICE void operator()(T& x)
  {
    allocator_traits<Allocator>::construct(a, &x, arg);
  }
};
template <typename Allocator, typename Pointer, typename Size, typename T>
_CCCL_HOST_DEVICE void fill_construct_range(Allocator& a, Pointer p, Size n, const T& value)
{
  if constexpr (has_effectful_member_construct2<Allocator, typename pointer_element<Pointer>::type, T>)
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a), p, n, construct2_via_allocator<Allocator, T>{a, value});
  }
  else
  {
    thrust::uninitialized_fill_n(allocator_system<Allocator>::get(a), p, n, value);
  }
}
} // namespace detail
THRUST_NAMESPACE_END
