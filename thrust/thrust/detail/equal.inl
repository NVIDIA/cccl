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
#include <thrust/equal.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/equal.h>
#include <thrust/system/detail/sequential/equal.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(equal.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(equal.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/equal.h>
#  include <thrust/system/cuda/detail/equal.h>
#  include <thrust/system/omp/detail/equal.h>
#  include <thrust/system/tbb/detail/equal.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename System, typename InputIterator1, typename InputIterator2>
_CCCL_HOST_DEVICE bool
equal(const thrust::detail::execution_policy_base<System>& system,
      InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::equal");
  using thrust::system::detail::generic::equal;
  return equal(thrust::detail::derived_cast(thrust::detail::strip_const(system)), first1, last1, first2);
} // end equal()

_CCCL_EXEC_CHECK_DISABLE
template <typename System, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE bool
equal(const thrust::detail::execution_policy_base<System>& system,
      InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2,
      BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::equal");
  using thrust::system::detail::generic::equal;
  return equal(thrust::detail::derived_cast(thrust::detail::strip_const(system)), first1, last1, first2, binary_pred);
} // end equal()

template <typename InputIterator1, typename InputIterator2>
bool equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::equal");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::equal(select_system(system1, system2), first1, last1, first2);
}

template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
bool equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::equal");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::equal(select_system(system1, system2), first1, last1, first2, binary_pred);
}

THRUST_NAMESPACE_END
