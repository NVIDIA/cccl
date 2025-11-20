/*
 *  Copyright 2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file find_end.h
 *  \brief Locating the last occurrence of a subsequence in a range
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

/*! \addtogroup algorithms
 */

/*! \addtogroup searching
 *  \ingroup algorithms
 *  \{
 */

/*! \p find_end finds the last occurrence of the subsequence <tt>[first2, last2)</tt>
 *  in the range <tt>[first1, last1)</tt>. Specifically, \p find_end returns the last
 *  iterator \c i in <tt>[first1, last1 - (last2 - first2))</tt> such that for every
 *  iterator \c j in <tt>[first2, last2)</tt>, <tt>*(i + (j - first2)) == *j</tt>.
 *  If no such iterator exists, \c last1 is returned.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the sequence to search.
 *  \param last1 The end of the sequence to search.
 *  \param first2 The beginning of the subsequence to find.
 *  \param last2 The end of the subsequence to find.
 *  \return The last iterator \c i in <tt>[first1, last1 - (last2 - first2))</tt>
 *          such that the subsequence <tt>[first2, last2)</tt> matches
 *          <tt>[i, i + (last2 - first2))</tt>, or \c last1 if no such iterator exists.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator1's \c value_type is equality comparable to \p ForwardIterator2's \c value_type.
 *
 *  \code
 *  #include <thrust/find_end.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *
 *  thrust::device_vector<int> data(10);
 *  data[0] = 0; data[1] = 1; data[2] = 2;
 *  data[3] = 0; data[4] = 1; data[5] = 2;
 *  data[6] = 3; data[7] = 4; data[8] = 5; data[9] = 6;
 *
 *  thrust::device_vector<int> pattern(3);
 *  pattern[0] = 0; pattern[1] = 1; pattern[2] = 2;
 *
 *  // Find the last occurrence of pattern in data
 *  thrust::device_vector<int>::iterator iter;
 *  iter = thrust::find_end(thrust::device, data.begin(), data.end(),
 *                          pattern.begin(), pattern.end());
 *
 *  // iter points to data[3] (the last occurrence starting at index 3)
 *  \endcode
 *
 *  \see find
 *  \see find_if
 *  \see mismatch
 */
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
_CCCL_HOST_DEVICE ForwardIterator1 find_end(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2,
  ForwardIterator2 last2);

/*! \p find_end finds the last occurrence of the subsequence <tt>[first2, last2)</tt>
 *  in the range <tt>[first1, last1)</tt>. Specifically, \p find_end returns the last
 *  iterator \c i in <tt>[first1, last1 - (last2 - first2))</tt> such that for every
 *  iterator \c j in <tt>[first2, last2)</tt>, <tt>pred(*(i + (j - first2)), *j)</tt>
 *  is \c true. If no such iterator exists, \c last1 is returned.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the sequence to search.
 *  \param last1 The end of the sequence to search.
 *  \param first2 The beginning of the subsequence to find.
 *  \param last2 The end of the subsequence to find.
 *  \param pred The binary predicate used for comparison.
 *  \return The last iterator \c i in <tt>[first1, last1 - (last2 - first2))</tt>
 *          such that the subsequence <tt>[first2, last2)</tt> matches
 *          <tt>[i, i + (last2 - first2))</tt> according to \p pred,
 *          or \c last1 if no such iterator exists.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary
 * Predicate</a>.
 *
 *  \code
 *  #include <thrust/find_end.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *
 *  struct compare_modulo_2
 *  {
 *    __host__ __device__
 *    bool operator()(int a, int b) const
 *    {
 *      return (a % 2) == (b % 2);
 *    }
 *  };
 *
 *  thrust::device_vector<int> data(10);
 *  data[0] = 0; data[1] = 1; data[2] = 2;
 *  data[3] = 2; data[4] = 3; data[5] = 4;
 *  data[6] = 3; data[7] = 4; data[8] = 5; data[9] = 6;
 *
 *  thrust::device_vector<int> pattern(3);
 *  pattern[0] = 0; pattern[1] = 1; pattern[2] = 2;
 *
 *  // Find the last occurrence where elements match modulo 2
 *  thrust::device_vector<int>::iterator iter;
 *  iter = thrust::find_end(thrust::device, data.begin(), data.end(),
 *                          pattern.begin(), pattern.end(),
 *                          compare_modulo_2());
 *
 *  // iter points to data[3] (elements at [3,4,5] are [2,3,4] which match [0,1,2] modulo 2)
 *  \endcode
 *
 *  \see find
 *  \see find_if
 *  \see mismatch
 */
template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator1 find_end(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2,
  ForwardIterator2 last2,
  BinaryPredicate pred);

/*! \p find_end finds the last occurrence of the subsequence <tt>[first2, last2)</tt>
 *  in the range <tt>[first1, last1)</tt>. Specifically, \p find_end returns the last
 *  iterator \c i in <tt>[first1, last1 - (last2 - first2))</tt> such that for every
 *  iterator \c j in <tt>[first2, last2)</tt>, <tt>*(i + (j - first2)) == *j</tt>.
 *  If no such iterator exists, \c last1 is returned.
 *
 *  \param first1 The beginning of the sequence to search.
 *  \param last1 The end of the sequence to search.
 *  \param first2 The beginning of the subsequence to find.
 *  \param last2 The end of the subsequence to find.
 *  \return The last iterator \c i in <tt>[first1, last1 - (last2 - first2))</tt>
 *          such that the subsequence <tt>[first2, last2)</tt> matches
 *          <tt>[i, i + (last2 - first2))</tt>, or \c last1 if no such iterator exists.
 *
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator1's \c value_type is equality comparable to \p ForwardIterator2's \c value_type.
 *
 *  \code
 *  #include <thrust/find_end.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> data(10);
 *  data[0] = 0; data[1] = 1; data[2] = 2;
 *  data[3] = 0; data[4] = 1; data[5] = 2;
 *  data[6] = 3; data[7] = 4; data[8] = 5; data[9] = 6;
 *
 *  thrust::device_vector<int> pattern(3);
 *  pattern[0] = 0; pattern[1] = 1; pattern[2] = 2;
 *
 *  // Find the last occurrence of pattern in data
 *  thrust::device_vector<int>::iterator iter;
 *  iter = thrust::find_end(data.begin(), data.end(),
 *                          pattern.begin(), pattern.end());
 *
 *  // iter points to data[3] (the last occurrence starting at index 3)
 *  \endcode
 *
 *  \see find
 *  \see find_if
 *  \see mismatch
 */
template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator1
find_end(ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2, ForwardIterator2 last2);

/*! \p find_end finds the last occurrence of the subsequence <tt>[first2, last2)</tt>
 *  in the range <tt>[first1, last1)</tt>. Specifically, \p find_end returns the last
 *  iterator \c i in <tt>[first1, last1 - (last2 - first2))</tt> such that for every
 *  iterator \c j in <tt>[first2, last2)</tt>, <tt>pred(*(i + (j - first2)), *j)</tt>
 *  is \c true. If no such iterator exists, \c last1 is returned.
 *
 *  \param first1 The beginning of the sequence to search.
 *  \param last1 The end of the sequence to search.
 *  \param first2 The beginning of the subsequence to find.
 *  \param last2 The end of the subsequence to find.
 *  \param pred The binary predicate used for comparison.
 *  \return The last iterator \c i in <tt>[first1, last1 - (last2 - first2))</tt>
 *          such that the subsequence <tt>[first2, last2)</tt> matches
 *          <tt>[i, i + (last2 - first2))</tt> according to \p pred,
 *          or \c last1 if no such iterator exists.
 *
 *  \tparam ForwardIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam ForwardIterator2 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/forward_iterator">Forward
 * Iterator</a>.
 *  \tparam BinaryPredicate is a model of <a href="https://en.cppreference.com/w/cpp/named_req/BinaryPredicate">Binary
 * Predicate</a>.
 *
 *  \code
 *  #include <thrust/find_end.h>
 *  #include <thrust/device_vector.h>
 *
 *  struct compare_modulo_2
 *  {
 *    __host__ __device__
 *    bool operator()(int a, int b) const
 *    {
 *      return (a % 2) == (b % 2);
 *    }
 *  };
 *
 *  ...
 *  thrust::device_vector<int> data(10);
 *  data[0] = 0; data[1] = 1; data[2] = 2;
 *  data[3] = 2; data[4] = 3; data[5] = 4;
 *  data[6] = 3; data[7] = 4; data[8] = 5; data[9] = 6;
 *
 *  thrust::device_vector<int> pattern(3);
 *  pattern[0] = 0; pattern[1] = 1; pattern[2] = 2;
 *
 *  // Find the last occurrence where elements match modulo 2
 *  thrust::device_vector<int>::iterator iter;
 *  iter = thrust::find_end(data.begin(), data.end(),
 *                          pattern.begin(), pattern.end(),
 *                          compare_modulo_2());
 *
 *  // iter points to data[3] (elements at [3,4,5] are [2,3,4] which match [0,1,2] modulo 2)
 *  \endcode
 *
 *  \see find
 *  \see find_if
 *  \see mismatch
 */
template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryPredicate>
ForwardIterator1 find_end(
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2,
  ForwardIterator2 last2,
  BinaryPredicate pred);

/*! \} // end searching
 */

THRUST_NAMESPACE_END

#include <thrust/detail/find_end.inl>
