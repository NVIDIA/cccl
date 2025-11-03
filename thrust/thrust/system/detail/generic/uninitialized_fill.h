// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
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
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/type_traits.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
namespace detail
{

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void uninitialized_fill(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const T& x,
  thrust::detail::true_type) // ::cuda::std::is_trivially_copy_constructible
{
  thrust::fill(exec, first, last, x);
} // end uninitialized_fill()

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void uninitialized_fill(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const T& x,
  thrust::detail::false_type) // ::cuda::std::is_trivially_copy_constructible
{
  using ValueType = thrust::detail::it_value_t<ForwardIterator>;

  thrust::for_each(exec, first, last, thrust::detail::uninitialized_fill_functor<ValueType>{x});
} // end uninitialized_fill()

template <typename DerivedPolicy, typename ForwardIterator, typename Size, typename T>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_fill_n(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  Size n,
  const T& x,
  thrust::detail::true_type) // ::cuda::std::is_trivially_copy_constructible
{
  return thrust::fill_n(exec, first, n, x);
} // end uninitialized_fill()

template <typename DerivedPolicy, typename ForwardIterator, typename Size, typename T>
_CCCL_HOST_DEVICE ForwardIterator uninitialized_fill_n(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  Size n,
  const T& x,
  thrust::detail::false_type) // ::cuda::std::is_trivially_copy_constructible
{
  using ValueType = thrust::detail::it_value_t<ForwardIterator>;

  return thrust::for_each_n(exec, first, n, thrust::detail::uninitialized_fill_functor<ValueType>{x});
} // end uninitialized_fill()

} // namespace detail

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void uninitialized_fill(
  thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, const T& x)
{
  using ValueType = thrust::detail::it_value_t<ForwardIterator>;

  using ValueTypeHasTrivialCopyConstructor = ::cuda::std::is_trivially_copy_constructible<ValueType>;

  thrust::system::detail::generic::detail::uninitialized_fill(
    exec, first, last, x, ValueTypeHasTrivialCopyConstructor());
} // end uninitialized_fill()

template <typename DerivedPolicy, typename ForwardIterator, typename Size, typename T>
_CCCL_HOST_DEVICE ForwardIterator
uninitialized_fill_n(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, Size n, const T& x)
{
  using ValueType = thrust::detail::it_value_t<ForwardIterator>;

  using ValueTypeHasTrivialCopyConstructor = ::cuda::std::is_trivially_copy_constructible<ValueType>;

  return thrust::system::detail::generic::detail::uninitialized_fill_n(
    exec, first, n, x, ValueTypeHasTrivialCopyConstructor());
} // end uninitialized_fill()

} // namespace system::detail::generic
THRUST_NAMESPACE_END
