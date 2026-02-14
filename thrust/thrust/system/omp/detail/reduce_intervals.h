// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file reduce_intervals.h
 *  \brief OpenMP implementations of reduce_intervals algorithms.
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
#include <thrust/detail/static_assert.h> // for depend_on_instantiation
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/omp/detail/pragma_omp.h>

#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(
  execution_policy<DerivedPolicy>&,
  InputIterator input,
  OutputIterator output,
  BinaryFunction binary_op,
  Decomposition decomp)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  static_assert(
    thrust::detail::depend_on_instantiation<InputIterator, (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value,
    "OpenMP compiler support is not enabled");

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  using OutputType = thrust::detail::it_value_t<OutputIterator>;

  // wrap binary_op
  thrust::detail::wrapped_function<BinaryFunction, OutputType> wrapped_binary_op{binary_op};

  using index_type = std::intptr_t;

  index_type n = static_cast<index_type>(decomp.size());

  THRUST_PRAGMA_OMP(parallel for)
  for (index_type i = 0; i < n; i++)
  {
    InputIterator begin = input + decomp[i].begin();
    InputIterator end   = input + decomp[i].end();

    if (begin != end)
    {
      OutputType sum = thrust::raw_reference_cast(*begin);

      ++begin;

      while (begin != end)
      {
        sum = wrapped_binary_op(sum, *begin);
        ++begin;
      }

      OutputIterator tmp = output + i;
      *tmp               = sum;
    }
  }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
}
} // end namespace system::omp::detail
THRUST_NAMESPACE_END
