// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file default_decomposition.h
 *  \brief Return a decomposition that is appropriate for the OpenMP backend.
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
#include <thrust/system/detail/internal/decompose.h>

// don't attempt to #include this file without omp support
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#  include <omp.h>
#endif // omp support

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
template <typename IndexType>
thrust::system::detail::internal::uniform_decomposition<IndexType> default_decomposition(IndexType n)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to OpenMP support in your compiler.                         X
  // ========================================================================
  static_assert(
    thrust::detail::depend_on_instantiation<IndexType, (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)>::value,
    "OpenMP compiler support is not enabled");

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  return thrust::system::detail::internal::uniform_decomposition<IndexType>(n, 1, omp_get_num_procs());
#else
  return thrust::system::detail::internal::uniform_decomposition<IndexType>(n, 1, 1);
#endif
}
} // end namespace system::omp::detail
THRUST_NAMESPACE_END
