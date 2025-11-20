// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file reduce.h
 *  \brief OpenMP implementation of reduce algorithms.
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

#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/omp/detail/default_decomposition.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/omp/detail/reduce_intervals.h>

#include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
template <typename DerivedPolicy, typename InputIterator, typename OutputType, typename BinaryFunction>
OutputType reduce(execution_policy<DerivedPolicy>& exec,
                  InputIterator first,
                  InputIterator last,
                  OutputType init,
                  BinaryFunction binary_op)
{
  using difference_type = thrust::detail::it_difference_t<InputIterator>;

  const difference_type n = ::cuda::std::distance(first, last);

  // determine first and second level decomposition
  thrust::system::detail::internal::uniform_decomposition<difference_type> decomp1 =
    thrust::system::omp::detail::default_decomposition(n);
  thrust::system::detail::internal::uniform_decomposition<difference_type> decomp2(decomp1.size() + 1, 1, 1);

  // allocate storage for the initializer and partial sums
  // XXX use select_system for Tag
  thrust::detail::temporary_array<OutputType, DerivedPolicy> partial_sums(exec, decomp1.size() + 1);

  // set first element of temp array to init
  partial_sums[0] = init;

  // accumulate partial sums (first level reduction)
  thrust::system::omp::detail::reduce_intervals(exec, first, partial_sums.begin() + 1, binary_op, decomp1);

  // reduce partial sums (second level reduction)
  thrust::system::omp::detail::reduce_intervals(exec, partial_sums.begin(), partial_sums.begin(), binary_op, decomp2);

  return partial_sums[0];
} // end reduce()
} // end namespace system::omp::detail
THRUST_NAMESPACE_END
