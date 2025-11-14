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
#include <thrust/detail/copy.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cuda/std/__cccl/memory_wrapper.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename Allocator, typename InputType, typename OutputType>
struct copy_construct_with_allocator
{
  Allocator& a;

  template <typename Tuple>
  inline _CCCL_HOST_DEVICE void operator()(Tuple t)
  {
    const InputType& in = thrust::get<0>(t);
    OutputType& out     = thrust::get<1>(t);

    allocator_traits<Allocator>::construct(a, &out, in);
  }
};

// we need to use allocator_traits<Allocator>::construct() to
// copy construct a T if either:
// 1. Allocator has a 2-argument construct() member or
// 2. T has a non-trivial copy constructor
template <typename Allocator, typename T>
inline constexpr bool needs_copy_construct_via_allocator =
  allocator_traits_detail::has_member_construct2<Allocator, T, T>::value
  || !::cuda::std::is_trivially_copy_constructible_v<T>;

// we know that std::allocator::construct's only effect is to call T's
// copy constructor, so we needn't consider or use its construct() member for copy construction
template <typename U, typename T>
inline constexpr bool needs_copy_construct_via_allocator<std::allocator<U>, T> =
  !::cuda::std::is_trivially_copy_constructible_v<T>;

// XXX it's regrettable that this implementation is copied almost
//     exactly from system::detail::generic::uninitialized_copy
//     perhaps generic::uninitialized_copy could call this routine
//     with a default allocator
template <typename Allocator, typename FromSystem, typename ToSystem, typename InputIterator, typename Pointer>
_CCCL_HOST_DEVICE Pointer uninitialized_copy_with_allocator(
  Allocator& a,
  const thrust::execution_policy<FromSystem>& from_system,
  const thrust::execution_policy<ToSystem>& to_system,
  InputIterator first,
  InputIterator last,
  Pointer result)
{
  if constexpr (::cuda::std::is_convertible_v<FromSystem, ToSystem>)
  {
    // zip up the iterators
    using IteratorTuple = thrust::tuple<InputIterator, Pointer>;
    using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

    ZipIterator begin = thrust::make_zip_iterator(first, result);
    ZipIterator end   = begin;

    // get a zip_iterator pointing to the end
    const thrust::detail::it_difference_t<InputIterator> n = ::cuda::std::distance(first, last);
    ::cuda::std::advance(end, n);

    // create a functor
    using InputType  = it_value_t<InputIterator>;
    using OutputType = it_value_t<Pointer>;

    // do the for_each
    // note we use to_system to dispatch the for_each
    thrust::for_each(to_system, begin, end, copy_construct_with_allocator<Allocator, InputType, OutputType>{a});

    // return the end of the output range
    return thrust::get<1>(end.get_iterator_tuple());
  }
  else
  {
    // the systems aren't trivially interoperable
    // just call two_system_copy and hope for the best
    return thrust::detail::two_system_copy(from_system, to_system, first, last, result);
  }
}

// XXX it's regrettable that this implementation is copied almost
//     exactly from system::detail::generic::uninitialized_copy_n
//     perhaps generic::uninitialized_copy_n could call this routine
//     with a default allocator
template <typename Allocator, typename FromSystem, typename ToSystem, typename InputIterator, typename Size, typename Pointer>
_CCCL_HOST_DEVICE Pointer uninitialized_copy_with_allocator_n(
  Allocator& a,
  const thrust::execution_policy<FromSystem>& from_system,
  const thrust::execution_policy<ToSystem>& to_system,
  InputIterator first,
  Size n,
  Pointer result)
{
  if constexpr (::cuda::std::is_convertible_v<FromSystem, ToSystem>)
  {
    // zip up the iterators
    using IteratorTuple = thrust::tuple<InputIterator, Pointer>;
    using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

    ZipIterator begin = thrust::make_zip_iterator(first, result);

    // create a functor
    using InputType  = it_value_t<InputIterator>;
    using OutputType = it_value_t<Pointer>;

    // do the for_each_n
    // note we use to_system to dispatch the for_each_n
    ZipIterator end =
      thrust::for_each_n(to_system, begin, n, copy_construct_with_allocator<Allocator, InputType, OutputType>(a));

    // return the end of the output range
    return thrust::get<1>(end.get_iterator_tuple());
  }
  else
  {
    // the systems aren't trivially interoperable
    // just call two_system_copy_n and hope for the best
    return thrust::detail::two_system_copy_n(from_system, to_system, first, n, result);
  }
}

template <typename System, typename Allocator, typename InputIterator, typename Pointer>
_CCCL_HOST_DEVICE Pointer copy_construct_range(
  thrust::execution_policy<System>& from_system, Allocator& a, InputIterator first, InputIterator last, Pointer result)
{
  if constexpr (needs_copy_construct_via_allocator<Allocator, typename pointer_element<Pointer>::type>)
  {
    return uninitialized_copy_with_allocator(a, from_system, allocator_system<Allocator>::get(a), first, last, result);
  }
  else
  {
    // just call two_system_copy
    return thrust::detail::two_system_copy(from_system, allocator_system<Allocator>::get(a), first, last, result);
  }
}

template <typename System, typename Allocator, typename InputIterator, typename Size, typename Pointer>
_CCCL_HOST_DEVICE Pointer copy_construct_range_n(
  thrust::execution_policy<System>& from_system, Allocator& a, InputIterator first, Size n, Pointer result)
{
  if constexpr (needs_copy_construct_via_allocator<Allocator, typename pointer_element<Pointer>::type>)
  {
    return uninitialized_copy_with_allocator_n(a, from_system, allocator_system<Allocator>::get(a), first, n, result);
  }
  else
  {
    // just call two_system_copy_n
    return thrust::detail::two_system_copy_n(from_system, allocator_system<Allocator>::get(a), first, n, result);
  }
}
} // namespace detail
THRUST_NAMESPACE_END
