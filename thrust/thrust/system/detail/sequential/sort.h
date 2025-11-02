/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

/*! \file sort.h
 *  \brief Sequential implementations of sort algorithms.
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

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/reverse.h>
#include <thrust/system/detail/sequential/execution_policy.h>
#include <thrust/system/detail/sequential/stable_merge_sort.h>
#include <thrust/system/detail/sequential/stable_primitive_sort.h>

#include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
namespace sort_detail
{
template <typename KeyType, typename Compare>
inline constexpr bool use_primitive_sort =
  ::cuda::std::is_arithmetic_v<KeyType>
  && (::cuda::std::is_same_v<Compare, ::cuda::std::less<KeyType>>
      || ::cuda::std::is_same_v<Compare, ::cuda::std::greater<KeyType>>);

template <typename KeyType, typename Compare>
inline constexpr bool needs_reverse = ::cuda::std::is_same_v<Compare, ::cuda::std::greater<KeyType>>;
} // end namespace sort_detail

template <typename DerivedPolicy, typename RandomAccessIterator, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void stable_sort(
  sequential::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator first,
  RandomAccessIterator last,
  [[maybe_unused]] StrictWeakOrdering comp) // GCC 7-9 warn that comp is unused
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      using KeyType = thrust::detail::it_value_t<RandomAccessIterator>;
      if constexpr (sort_detail::use_primitive_sort<KeyType, StrictWeakOrdering>) {
        thrust::system::detail::sequential::stable_primitive_sort(exec, first, last);

        // if comp is greater<T> then reverse the keys
        if constexpr (sort_detail::needs_reverse<KeyType, StrictWeakOrdering>)
        {
          thrust::reverse(exec, first, last);
        }
      } else { thrust::system::detail::sequential::stable_merge_sort(exec, first, last, comp); }),
    ( // NV_IS_DEVICE:
      // the compilation time of stable_primitive_sort is too expensive to use within a single CUDA thread
      thrust::system::detail::sequential::stable_merge_sort(exec, first, last, comp);));
#if _CCCL_COMPILER(GCC, <, 10)
  (void) comp; // GCC 7-9 warn that comp is unused
#endif // _CCCL_COMPILER(GCC, <, 10)
}

template <typename DerivedPolicy,
          typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE void stable_sort_by_key(
  sequential::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 first1,
  RandomAccessIterator1 last1,
  RandomAccessIterator2 first2,
  [[maybe_unused]] StrictWeakOrdering comp) // GCC 7-9 warn that comp is unused
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      using KeyType = thrust::detail::it_value_t<RandomAccessIterator1>;
      if constexpr (sort_detail::use_primitive_sort<KeyType, StrictWeakOrdering>) {
        // if comp is greater<T> then reverse the keys and values
        // note, we also have to reverse the (unordered) input to preserve stability
        if constexpr (sort_detail::needs_reverse<KeyType, StrictWeakOrdering>)
        {
          thrust::reverse(exec, first1, last1);
          thrust::reverse(exec, first2, first2 + (last1 - first1));
        }
        thrust::system::detail::sequential::stable_primitive_sort_by_key(exec, first1, last1, first2);
        if constexpr (sort_detail::needs_reverse<KeyType, StrictWeakOrdering>)
        {
          thrust::reverse(exec, first1, last1);
          thrust::reverse(exec, first2, first2 + (last1 - first1));
        }
      } else { thrust::system::detail::sequential::stable_merge_sort_by_key(exec, first1, last1, first2, comp); }),
    ( // NV_IS_DEVICE:
      // the compilation time of stable_primitive_sort is too expensive to use within a single CUDA thread
      thrust::system::detail::sequential::stable_merge_sort_by_key(exec, first1, last1, first2, comp);));
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
