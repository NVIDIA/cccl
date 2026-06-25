// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file thrust/erase.h
 *  \brief TODO
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
#include <thrust/detail/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>

#include <type_traits>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup copying
 *  \ingroup algorithms
 *  \{
 */

// TODO: FIx documentation
/*! \p copy copies elements from the range [\p first, \p last) to the range
 *  [\p result, \p result + (\p last - \p first)). That is, it performs
 *  the assignments *\p result = *\p first, *(\p result + \c 1) = *(\p first + \c 1),
 *  and so on. Generally, for every integer \c n from \c 0 to \p last - \p first, \p copy
 *  performs the assignment *(\p result + \c n) = *(\p first + \c n). Unlike
 *  \c std::copy, \p copy offers no guarantee on order of operation.  As a result,
 *  calling \p copy with overlapping source and destination ranges has undefined
 *  behavior.
 *
 *  The return value is \p result + (\p last - \p first).
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence to copy.
 *  \param last The end of the sequence to copy.
 *  \param result The destination sequence.
 *  \return The end of the destination sequence.
 *  \see https://en.cppreference.com/w/cpp/algorithm/copy
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input
 * Iterator</a> and \c InputIterator's \c value_type must be convertible to \c OutputIterator's \c value_type. \tparam
 * OutputIterator must be a model of <a href="https://en.cppreference.com/w/cpp/iterator/output_iterator">Output
 * Iterator</a>.
 *
 *  \pre \p result may be equal to \p first, but \p result shall not be in the range <tt>[first, last)</tt> otherwise.
 *
 *  The following code snippet demonstrates how to use \p copy
 *  to copy from one range to another using the \p thrust::device parallelization policy:
 *
 *  \code
 *  #include <thrust/copy.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  thrust::device_vector<int> vec0(100);
 *  thrust::device_vector<int> vec1(100);
 *  ...
 *
 *  thrust::copy(thrust::device, vec0.begin(), vec0.end(), vec1.begin());
 *
 *  // vec1 is now a copy of vec0
 *  \endcode
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: 2.2.0
 *  \endverbatim
 */

template <class V>
struct is_thrust_vector : std::false_type
{};

template <class T, class Alloc>
struct is_thrust_vector<thrust::host_vector<T, Alloc>> : std::true_type
{};

template <class T, class Alloc>
struct is_thrust_vector<thrust::device_vector<T, Alloc>> : std::true_type
{};

// Add exec policy function alternative

template <class Vector,
          class U                                                        = typename Vector::value_type,
          ::cuda::std::enable_if_t<is_thrust_vector<Vector>::value, int> = 0>
typename Vector::size_type erase(Vector& c, const U& value)
{
  using value_type = typename Vector::value_type;

  auto first = thrust::remove(c.begin(), c.end(), static_cast<value_type>(value));

  auto removed = static_cast<typename Vector::size_type>(::cuda::std::distance(first, c.end()));

  c.erase(first, c.end());

  return removed;
}

THRUST_NAMESPACE_END
