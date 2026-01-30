// SPDX-FileCopyrightText: Copyright (c) 2008-2020, NVIDIA Corporation. All rights reserved.
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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/shuffle.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/shuffle.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename RandomIterator, typename URBG>
_CCCL_HOST_DEVICE void shuffle(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec, RandomIterator first, RandomIterator last, URBG&& g)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::shuffle");
  using thrust::system::detail::generic::shuffle;
  return shuffle(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, g);
}

template <typename RandomIterator, typename URBG>
_CCCL_HOST_DEVICE void shuffle(RandomIterator first, RandomIterator last, URBG&& g)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::shuffle");
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<RandomIterator>::type;
  System system;

  return thrust::shuffle(select_system(system), first, last, g);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename RandomIterator, typename OutputIterator, typename URBG>
_CCCL_HOST_DEVICE void shuffle_copy(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  RandomIterator first,
  RandomIterator last,
  OutputIterator result,
  URBG&& g)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::shuffle_copy");
  using thrust::system::detail::generic::shuffle_copy;
  return shuffle_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, g);
}

template <typename RandomIterator, typename OutputIterator, typename URBG>
_CCCL_HOST_DEVICE void shuffle_copy(RandomIterator first, RandomIterator last, OutputIterator result, URBG&& g)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::shuffle_copy");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<RandomIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::shuffle_copy(select_system(system1, system2), first, last, result, g);
}

THRUST_NAMESPACE_END
