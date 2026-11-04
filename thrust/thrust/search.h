// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file search.h
 *  \brief Search for the first occurrence of a subsequence within a range
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

THRUST_NAMESPACE_BEGIN

/*! \addtogroup searching
 *  \{
 */

/*! \p search finds the first occurrence of the subsequence <tt>[s_first, s_last)</tt>
 *  in the range <tt>[first, last)</tt>. The algorithm compares elements using
 *  <tt>operator==</tt>.
 *
 *  Specifically, \p search returns the first iterator \c i in <tt>[first, last - (s_last - s_first))</tt>
 *  such that, for every iterator \c j in <tt>[s_first, s_last)</tt>,
 *  <tt>*(i + (j - s_first)) == *j</tt>. If no such iterator exists, \p search returns \p last.
 *
 *  \param first The beginning of the range to search.
 *  \param last The end of the range to search.
 *  \param s_first The beginning of the subsequence to find.
 *  \param s_last The end of the subsequence to find.
 *  \return An iterator pointing to the beginning of the first occurrence of the subsequence
 *          <tt>[s_first, s_last)</tt> in <tt>[first, last)</tt>. If the subsequence is not found,
 *          returns \p last. If <tt>[s_first, s_last)</tt> is empty, returns \p first.
 *
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator1's value type is equality comparable to ForwardIterator2's value type.
 *
 *  The following code snippet demonstrates how to use \p search to find the first
 *  occurrence of a subsequence in a range.
 *
 *  \code
 *  #include <thrust/search.h>
 *  #include <thrust/device_vector.h>
 *
 *  thrust::device_vector<int> data{0, 1, 2, 3, 4, 1, 2, 5};
 *  thrust::device_vector<int> pattern{1, 2};
 *
 *  // Find first occurrence of {1, 2}
 *  auto iter = thrust::search(data.begin(), data.end(), pattern.begin(), pattern.end());
 *  // iter points to data.begin() + 1 (first occurrence)
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/search
 *  \see find
 *  \see find_if
 *  \see mismatch
 */
template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator1
search(ForwardIterator1 first, ForwardIterator1 last, ForwardIterator2 s_first, ForwardIterator2 s_last);

/*! \p search finds the first occurrence of the subsequence <tt>[s_first, s_last)</tt>
 *  in the range <tt>[first, last)</tt>. The algorithm compares elements using
 *  the function object \p pred.
 *
 *  Specifically, \p search returns the first iterator \c i in <tt>[first, last - (s_last - s_first))</tt>
 *  such that, for every iterator \c j in <tt>[s_first, s_last)</tt>,
 *  <tt>pred(*(i + (j - s_first)), *j)</tt> is \c true. If no such iterator exists,
 *  \p search returns \p last.
 *
 *  \param first The beginning of the range to search.
 *  \param last The end of the range to search.
 *  \param s_first The beginning of the subsequence to find.
 *  \param s_last The end of the subsequence to find.
 *  \param pred The binary predicate used for comparison.
 *  \return An iterator pointing to the beginning of the first occurrence of the subsequence
 *          <tt>[s_first, s_last)</tt> in <tt>[first, last)</tt>. If the subsequence is not found,
 *          returns \p last. If <tt>[s_first, s_last)</tt> is empty, returns \p first.
 *
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary
 * Predicate</a>.
 *
 *  The following code snippet demonstrates how to use \p search with a custom predicate.
 *
 *  \code
 *  #include <thrust/search.h>
 *  #include <thrust/device_vector.h>
 *
 *  struct compare_modulo_two
 *  {
 *    __host__ __device__
 *    bool operator()(int a, int b) const
 *    {
 *      return (a % 2) == (b % 2);
 *    }
 *  };
 *
 *  thrust::device_vector<int> data{0, 1, 3, 2, 4, 5, 7, 6};
 *  thrust::device_vector<int> pattern{1, 3}; // Two odd numbers
 *
 *  // Find first occurrence of two consecutive odd numbers
 *  auto iter = thrust::search(data.begin(), data.end(),
 *                             pattern.begin(), pattern.end(),
 *                             compare_modulo_two());
 *  // iter points to data.begin() + 1 (1, 3 are both odd)
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/algorithm/search
 *  \see find
 *  \see find_if
 *  \see mismatch
 */
template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
ForwardIterator1
search(ForwardIterator1 first,
       ForwardIterator1 last,
       ForwardIterator2 s_first,
       ForwardIterator2 s_last,
       BinaryPredicate pred);

/*! \p search finds the first occurrence of the subsequence <tt>[s_first, s_last)</tt>
 *  in the range <tt>[first, last)</tt> using the specified execution policy.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range to search.
 *  \param last The end of the range to search.
 *  \param s_first The beginning of the subsequence to find.
 *  \param s_last The end of the subsequence to find.
 *  \return An iterator pointing to the beginning of the first occurrence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 */
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ForwardIterator1 search(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 first,
  ForwardIterator1 last,
  ForwardIterator2 s_first,
  ForwardIterator2 s_last);

/*! \p search finds the first occurrence of the subsequence <tt>[s_first, s_last)</tt>
 *  in the range <tt>[first, last)</tt> using the specified execution policy and predicate.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the range to search.
 *  \param last The end of the range to search.
 *  \param s_first The beginning of the subsequence to find.
 *  \param s_last The end of the subsequence to find.
 *  \param pred The binary predicate used for comparison.
 *  \return An iterator pointing to the beginning of the first occurrence.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary
 * Predicate</a>.
 */
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator1 search(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 first,
  ForwardIterator1 last,
  ForwardIterator2 s_first,
  ForwardIterator2 s_last,
  BinaryPredicate pred);

/*! \} // end searching
 */

THRUST_NAMESPACE_END

#include <thrust/detail/search.inl>
