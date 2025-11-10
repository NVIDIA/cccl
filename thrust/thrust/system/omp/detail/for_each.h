// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file for_each.h
 *  \brief Defines the interface for a function that executes a
 *  function or functional for each value in a given range.
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

#include <thrust/detail/function.h>
#include <thrust/detail/static_assert.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/omp/detail/pragma_omp.h>

#include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
template <typename DerivedPolicy, typename RandomAccessIterator, typename Size, typename UnaryFunction>
RandomAccessIterator for_each_n(execution_policy<DerivedPolicy>&, RandomAccessIterator first, Size n, UnaryFunction f)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  static_assert(thrust::detail::depend_on_instantiation<RandomAccessIterator,
                                                        (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value,
                "OpenMP compiler support is not enabled");

  if (n <= 0)
  {
    return first; // empty range
  }

  // create a wrapped function for f
  thrust::detail::wrapped_function<UnaryFunction, void> wrapped_f{f};

  // use a signed type for the iteration variable or suffer the consequences of warnings
  using DifferenceType    = thrust::detail::it_difference_t<RandomAccessIterator>;
  DifferenceType signed_n = n;

  THRUST_PRAGMA_OMP(parallel for)
  for (DifferenceType i = 0; i < signed_n; ++i)
  {
    RandomAccessIterator temp = first + i;
    wrapped_f(*temp);
  }

  return first + n;
} // end for_each_n()

template <typename DerivedPolicy, typename RandomAccessIterator, typename UnaryFunction>
RandomAccessIterator
for_each(execution_policy<DerivedPolicy>& s, RandomAccessIterator first, RandomAccessIterator last, UnaryFunction f)
{
  return omp::detail::for_each_n(s, first, ::cuda::std::distance(first, last), f);
} // end for_each()
} // end namespace system::omp::detail
THRUST_NAMESPACE_END
