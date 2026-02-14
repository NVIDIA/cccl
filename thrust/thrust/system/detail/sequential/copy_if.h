// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file copy_if.h
 *  \brief Sequential implementation of copy_if.
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
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
_CCCL_HOST_DEVICE OutputIterator copy_if(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred)
{
  thrust::detail::wrapped_function<Predicate, bool> wrapped_pred{pred};

  while (first != last)
  {
    if (wrapped_pred(*stencil))
    {
      *result = *first;
      ++result;
    } // end if

    ++first;
    ++stencil;
  } // end while

  return result;
} // end copy_if()
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
