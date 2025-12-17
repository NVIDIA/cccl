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
#include <thrust/detail/allocator/destroy_range.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/for_each.h>

#include <cuda/std/__cccl/memory_wrapper.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// destroy_range has three cases:
// if Allocator has an effectful member function destroy:
//   1. destroy via the allocator
// else
//   2. if T has a non-trivial destructor, destroy the range without using the allocator
//   3. if T has a trivial destructor, do a no-op

template <typename Allocator, typename T>
inline constexpr bool has_effectful_member_destroy = allocator_traits_detail::has_member_destroy<Allocator, T>::value;

// std::allocator::destroy's only effect is to invoke its argument's destructor
template <typename U, typename T>
inline constexpr bool has_effectful_member_destroy<std::allocator<U>, T> = false;

template <typename Allocator>
struct destroy_via_allocator
{
  Allocator& a;

  template <typename T>
  _CCCL_HOST_DEVICE void operator()(T& x) noexcept
  {
    allocator_traits<Allocator>::destroy(a, &x);
  }
};

// we must prepare for His coming
struct gozer
{
  _CCCL_EXEC_CHECK_DISABLE
  template <typename T>
  inline _CCCL_HOST_DEVICE void operator()(T& x) noexcept
  {
    x.~T();
  }
};

template <typename Allocator, typename Pointer, typename Size>
_CCCL_HOST_DEVICE void
destroy_range([[maybe_unused]] Allocator& a, [[maybe_unused]] Pointer p, [[maybe_unused]] Size n) noexcept
{
  using pe_t = typename pointer_element<Pointer>::type;

  // case 1: destroy via allocator
  if constexpr (has_effectful_member_destroy<Allocator, pe_t>)
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a), p, n, destroy_via_allocator<Allocator>{a});
  }
  // case 2: destroy without the allocator
  else if constexpr (!::cuda::std::is_trivially_destructible_v<pe_t>)
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a), p, n, gozer());
  }
  // case 3: Allocator has no member function "destroy", and T has a trivial destructor, nothing to be done
}
} // namespace detail
THRUST_NAMESPACE_END
