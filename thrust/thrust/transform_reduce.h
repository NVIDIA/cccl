/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file transform_reduce.h
 *  \brief Fused transform / reduction
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

/*! \addtogroup reductions
 *  \{
 *  \addtogroup transformed_reductions Transformed Reductions
 *  \ingroup reductions
 *  \{
 */

/*! \p transform_reduce fuses the \p transform and \p reduce operations.
 *  \p transform_reduce is equivalent to performing a transformation defined by
 *  \p unary_op into a temporary sequence and then performing \p reduce on the
 *  transformed sequence. In most cases, fusing these two operations together is
 *  more efficient, since fewer memory reads and writes are required.
 *
 *  \p transform_reduce performs a reduction on the transformation of the
 *  sequence <tt>[first, last)</tt> according to \p unary_op. Specifically,
 *  \p unary_op is applied to each element of the sequence and then the result
 *  is reduced to a single value with \p binary_op using the initial value
 *  \p init.  Note that the transformation \p unary_op is not applied to
 *  the initial value \p init.  The order of reduction is not specified,
 *  so \p binary_op must be both commutative and associative.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param unary_op The function to apply to each element of the input sequence.
 *  \param init The result is initialized to this value.
 *  \param binary_op The reduction operation.
 *  \return The result of the transformed reduction.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input
 * Iterator</a>, and \p InputIterator's \c value_type is convertible to \p UnaryFunction's argument type.
 *  \tparam UnaryFunction The function's return type must be convertible to \c OutputType.
 *  \tparam OutputType is a model of
 * <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and is convertible to \p
 * BinaryFunction's first and second argument type.
 *  \tparam BinaryFunction The function's return type must be convertible to \p OutputType.
 *
 *  The following code snippet demonstrates how to use \p transform_reduce
 *  to compute the maximum value of the absolute value of the elements
 *  of a range using the \p thrust::host execution policy for parallelization:
 *
 *  \code
 *  #include <thrust/transform_reduce.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *
 *  template<typename T>
 *  struct absolute_value
 *  {
 *    __host__ __device__ T operator()(const T &x) const
 *    {
 *      return x < T(0) ? -x : x;
 *    }
 *  };
 *
 *  ...
 *
 *  int data[6] = {-1, 0, -2, -2, 1, -3};
 *  int result = thrust::transform_reduce(thrust::host,
 *                                        data, data + 6,
 *                                        absolute_value<int>(),
 *                                        0,
 *                                        ::cuda::maximum<int>());
 *  // result == 3
 *  \endcode
 *
 *  \see \c transform
 *  \see \c reduce
 */
template <typename DerivedPolicy,
          typename InputIterator,
          typename UnaryFunction,
          typename OutputType,
          typename BinaryFunction>
_CCCL_HOST_DEVICE OutputType transform_reduce(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  UnaryFunction unary_op,
  OutputType init,
  BinaryFunction binary_op);

/*! \p transform_reduce fuses the \p transform and \p reduce operations.
 *  \p transform_reduce is equivalent to performing a transformation defined by
 *  \p unary_op into a temporary sequence and then performing \p reduce on the
 *  transformed sequence. In most cases, fusing these two operations together is
 *  more efficient, since fewer memory reads and writes are required.
 *
 *  \p transform_reduce performs a reduction on the transformation of the
 *  sequence <tt>[first, last)</tt> according to \p unary_op. Specifically,
 *  \p unary_op is applied to each element of the sequence and then the result
 *  is reduced to a single value with \p binary_op using the initial value
 *  \p init.  Note that the transformation \p unary_op is not applied to
 *  the initial value \p init.  The order of reduction is not specified,
 *  so \p binary_op must be both commutative and associative.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param unary_op The function to apply to each element of the input sequence.
 *  \param init The result is initialized to this value.
 *  \param binary_op The reduction operation.
 *  \return The result of the transformed reduction.
 *
 *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input
 * Iterator</a>, and \p InputIterator's \c value_type is convertible to \p UnaryFunction's argument type.
 *  \tparam UnaryFunction The function's return type must be convertible to \c OutputType.
 *  \tparam OutputType is a model of
 * <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>, and is convertible to \p
 * BinaryFunction's first and second argument type.
 *  \tparam BinaryFunction The function's return type must be convertible to \p OutputType.
 *
 *  The following code snippet demonstrates how to use \p transform_reduce
 *  to compute the maximum value of the absolute value of the elements
 *  of a range.
 *
 *  \code
 *  #include <thrust/transform_reduce.h>
 *  #include <thrust/functional.h>
 *
 *  template<typename T>
 *  struct absolute_value
 *  {
 *    __host__ __device__ T operator()(const T &x) const
 *    {
 *      return x < T(0) ? -x : x;
 *    }
 *  };
 *
 *  ...
 *
 *  int data[6] = {-1, 0, -2, -2, 1, -3};
 *  int result = thrust::transform_reduce(data, data + 6,
 *                                        absolute_value<int>(),
 *                                        0,
 *                                        ::cuda::maximum<int>());
 *  // result == 3
 *  \endcode
 *
 *  \see \c transform
 *  \see \c reduce
 */
template <typename InputIterator, typename UnaryFunction, typename OutputType, typename BinaryFunction>
OutputType transform_reduce(
  InputIterator first, InputIterator last, UnaryFunction unary_op, OutputType init, BinaryFunction binary_op);

/*! \p transform_reduce fuses the \p transform and \p reduce operations on two input ranges.
 *  This version computes the generalized sum of <tt>binary_op2(*first1, *first2) + binary_op2(*(first1+1),
 * *(first2+1)) + ...</tt> with an initial value \p init.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input sequence.
 *  \param last1 The end of the first input sequence.
 *  \param first2 The beginning of the second input sequence.
 *  \param init The initial value of the reduction.
 *  \return The result of the transformed reduction: <tt>init + binary_op2(*first1, *first2) + ...</tt>
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input
 * Iterator</a>. \tparam InputIterator2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. \tparam T is convertible to the
 * return type of <tt>binary_op2(*first1, *first2)</tt> and supports the addition operation with the return type.
 *
 *  \code
 *  #include <thrust/transform_reduce.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int data1[6] = {1, 0, 2, 2, 1, 3};
 *  int data2[6] = {4, 1, 5, 3, 2, 1};
 *  int result = thrust::transform_reduce(thrust::host, data1, data1 + 6, data2, 0);
 *  // result == 1*4 + 0*1 + 2*5 + 2*3 + 1*2 + 3*1 = 25
 *  \endcode
 *
 *  \see \c transform
 *  \see \c reduce
 *  \see \c inner_product
 */
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename T>
_CCCL_HOST_DEVICE T transform_reduce(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  T init);

/*! \p transform_reduce fuses the \p transform and \p reduce operations on two input ranges.
 *  This version computes the generalized sum of <tt>binary_op2(*first1, *first2) + binary_op2(*(first1+1),
 * *(first2+1)) + ...</tt> with an initial value \p init.
 *
 *  \param first1 The beginning of the first input sequence.
 *  \param last1 The end of the first input sequence.
 *  \param first2 The beginning of the second input sequence.
 *  \param init The initial value of the reduction.
 *  \return The result of the transformed reduction: <tt>init + binary_op2(*first1, *first2) + ...</tt>
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input
 * Iterator</a>. \tparam InputIterator2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. \tparam T is convertible to the
 * return type of <tt>binary_op2(*first1, *first2)</tt> and supports the addition operation with the return type.
 *
 *  \code
 *  #include <thrust/transform_reduce.h>
 *  ...
 *  int data1[6] = {1, 0, 2, 2, 1, 3};
 *  int data2[6] = {4, 1, 5, 3, 2, 1};
 *  int result = thrust::transform_reduce(data1, data1 + 6, data2, 0);
 *  // result == 1*4 + 0*1 + 2*5 + 2*3 + 1*2 + 3*1 = 25
 *  \endcode
 *
 *  \see \c transform
 *  \see \c reduce
 *  \see \c inner_product
 */
template <typename InputIterator1, typename InputIterator2, typename T>
T transform_reduce(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, T init);

/*! \p transform_reduce fuses the \p transform and \p reduce operations on two input ranges
 *  with custom binary operations.
 *
 *  This version computes the generalized sum
 *  <tt>binary_op1(init, binary_op2(*first1, *first2)), binary_op1(..., binary_op2(*(first1+1), *(first2+1))), ...</tt>
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param first1 The beginning of the first input sequence.
 *  \param last1 The end of the first input sequence.
 *  \param first2 The beginning of the second input sequence.
 *  \param init The initial value of the reduction.
 *  \param binary_op1 The binary reduction operation (e.g., addition).
 *  \param binary_op2 The binary transformation operation (e.g., multiplication).
 *  \return The result of the transformed reduction.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input
 * Iterator</a>. \tparam InputIterator2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. \tparam T is convertible to \p
 * BinaryOp1's first and second argument types. \tparam BinaryOp1 is a binary function whose return type is convertible
 * to \p T. \tparam BinaryOp2 is a binary function whose return type is convertible to \p BinaryOp1's second argument
 * type.
 *
 *  \code
 *  #include <thrust/transform_reduce.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *  int data1[6] = {1, 0, 2, 2, 1, 3};
 *  int data2[6] = {4, 1, 5, 3, 2, 1};
 *  int result = thrust::transform_reduce(thrust::host, data1, data1 + 6, data2, 0,
 *                                        ::cuda::std::plus<int>(), ::cuda::std::multiplies<int>());
 *  // result == 0 + 1*4 + 0*1 + 2*5 + 2*3 + 1*2 + 3*1 = 25
 *  \endcode
 *
 *  \see \c transform
 *  \see \c reduce
 *  \see \c inner_product
 */
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename T,
          typename BinaryOp1,
          typename BinaryOp2>
_CCCL_HOST_DEVICE T transform_reduce(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  T init,
  BinaryOp1 binary_op1,
  BinaryOp2 binary_op2);

/*! \p transform_reduce fuses the \p transform and \p reduce operations on two input ranges
 *  with custom binary operations.
 *
 *  This version computes the generalized sum
 *  <tt>binary_op1(init, binary_op2(*first1, *first2)), binary_op1(..., binary_op2(*(first1+1), *(first2+1))), ...</tt>
 *
 *  \param first1 The beginning of the first input sequence.
 *  \param last1 The end of the first input sequence.
 *  \param first2 The beginning of the second input sequence.
 *  \param init The initial value of the reduction.
 *  \param binary_op1 The binary reduction operation (e.g., addition).
 *  \param binary_op2 The binary transformation operation (e.g., multiplication).
 *  \return The result of the transformed reduction.
 *
 *  \tparam InputIterator1 is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input
 * Iterator</a>. \tparam InputIterator2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/iterator/input_iterator">Input Iterator</a>. \tparam T is convertible to \p
 * BinaryOp1's first and second argument types. \tparam BinaryOp1 is a binary function whose return type is convertible
 * to \p T. \tparam BinaryOp2 is a binary function whose return type is convertible to \p BinaryOp1's second argument
 * type.
 *
 *  \code
 *  #include <thrust/transform_reduce.h>
 *  #include <thrust/functional.h>
 *  ...
 *  int data1[6] = {1, 0, 2, 2, 1, 3};
 *  int data2[6] = {4, 1, 5, 3, 2, 1};
 *  int result = thrust::transform_reduce(data1, data1 + 6, data2, 0,
 *                                        ::cuda::std::plus<int>(), ::cuda::std::multiplies<int>());
 *  // result == 0 + 1*4 + 0*1 + 2*5 + 2*3 + 1*2 + 3*1 = 25
 *  \endcode
 *
 *  \see \c transform
 *  \see \c reduce
 *  \see \c inner_product
 */
template <typename InputIterator1, typename InputIterator2, typename T, typename BinaryOp1, typename BinaryOp2>
T transform_reduce(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  T init,
  BinaryOp1 binary_op1,
  BinaryOp2 binary_op2);

/*! \} // end transformed_reductions
 *  \} // end reductions
 */

THRUST_NAMESPACE_END

#include <thrust/detail/transform_reduce.inl>
