// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file thrust/erase.h
 *  \brief Functions to erase and erase_if elements from Thrust vectors
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
#include <thrust/type_traits/is_thrust_vector.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup algorithms
 */

/*! \addtogroup copying
 *  \ingroup algorithms
 *  \{
 */

/*! \p erase removes all elements equal to \p value from the vector \p c.
 *  It performs the operation of erasing matching elements and shifting the remaining
 *  elements towards the beginning of the vector. The size of the vector is reduced
 *  by the number of erased elements.
 *
 *  The return value is the number of elements that were erased.
 *
 *  Note: unlike \p remove, erase performs container resizing and is host-only.
 *
 *  \param c The vector from which to erase elements.
 *  \param value The value to erase from the vector.
 *  \return The number of elements erased.
 *  \see https://en.cppreference.com/cpp/container/vector/erase2
 *
 *  \tparam Vector must be a Thrust vector type.
 *  \tparam U must be a type convertible to \c Vector's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p erase
 *  to remove all instances of a value from a vector:
 *
 *  \code
 *  #include <thrust/erase.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *
 *  thrust::device_vector<int> vec{1, 2, 3, 2, 4};
 *  ...
 *
 *  auto erased = thrust::erase(vec, 2);
 *
 *  // vec is now {1, 3, 4}
 *  // erased is 2
 *  \endcode
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: ?.?.?
 *  \endverbatim
 */
template <class Vector, class U, ::cuda::std::enable_if_t<is_thrust_vector_v<Vector>, int> = 0>
_CCCL_HOST typename Vector::size_type erase(Vector& c, const U& value);

/*! \p erase removes all elements equal to \p value from the vector \p c.
 *  It performs the operation of erasing matching elements and shifting the remaining
 *  elements towards the beginning of the vector. The size of the vector is reduced
 *  by the number of erased elements.
 *
 *  The return value is the number of elements that were erased.
 *
 *  Note: unlike \p remove, erase performs container resizing and is host-only.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param c The vector from which to erase elements.
 *  \param value The value to erase from the vector.
 *  \return The number of elements erased.
 *  \see https://en.cppreference.com/cpp/container/vector/erase2
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam Vector must be a Thrust vector type.
 *  \tparam U must be a type convertible to \c Vector's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p erase
 *  to remove all instances of a value from a vector using the \p thrust::device parallelization policy:
 *
 *  \code
 *  #include <thrust/erase.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  thrust::device_vector<int> vec{1, 2, 3, 2, 4};
 *  ...
 *
 *  auto erased = thrust::erase(thrust::device, vec, 2);
 *
 *  // vec is now {1, 3, 4}
 *  // erased is 2
 *  \endcode
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: ?.?.?
 *  \endverbatim
 */
template <typename DerivedPolicy, class Vector, class U, ::cuda::std::enable_if_t<is_thrust_vector_v<Vector>, int> = 0>
_CCCL_HOST typename Vector::size_type
erase(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Vector& c, const U& value);

/*! \p erase_if removes all elements from the vector \p c that satisfy the predicate \p pred.
 *  It performs the operation of erasing matching elements and shifting the remaining
 *  elements towards the beginning of the vector. The size of the vector is reduced
 *  by the number of erased elements.
 *
 *  The return value is the number of elements that were erased.
 *
 *  Note: unlike \p remove_if, erase_if performs container resizing and is host-only.
 *
 *  \param c The vector from which to erase elements.
 *  \param pred The predicate used to determine which elements to erase.
 *  \return The number of elements erased.
 *  \see https://en.cppreference.com/cpp/container/vector/erase2
 *
 *  \tparam Vector must be a Thrust vector type.
 *  \tparam Predicate must be a unary functor that returns \c true for elements to erase.
 *
 *  The following code snippet demonstrates how to use \p erase_if
 *  to remove all elements satisfying a predicate from a vector:
 *
 *  \code
 *  #include <thrust/erase.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *
 *  struct is_negative
 *  {
 *    __host__ __device__
 *    bool operator()(int x) { return x < 0; }
 *  };
 *
 *  thrust::device_vector<int> vec{1, -2, 3, -4, 5};
 *  ...
 *
 *  auto erased = thrust::erase_if(vec, is_negative());
 *
 *  // vec is now {1, 3, 5}
 *  // erased is 2
 *  \endcode
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: ?.?.?
 *  \endverbatim
 */
template <class Vector, class Predicate, ::cuda::std::enable_if_t<is_thrust_vector_v<Vector>, int> = 0>
_CCCL_HOST typename Vector::size_type erase_if(Vector& c, Predicate pred);

/*! \p erase_if removes all elements from the vector \p c that satisfy the predicate \p pred.
 *  It performs the operation of erasing matching elements and shifting the remaining
 *  elements towards the beginning of the vector. The size of the vector is reduced
 *  by the number of erased elements.
 *
 *  The return value is the number of elements that were erased.
 *
 *  Note: unlike \p remove_if, erase_if performs container resizing and is host-only.
 *
 *  The algorithm's execution is parallelized as determined by \p exec.
 *
 *  \param exec The execution policy to use for parallelization.
 *  \param c The vector from which to erase elements.
 *  \param pred The predicate used to determine which elements to erase.
 *  \return The number of elements erased.
 *  \see https://en.cppreference.com/cpp/container/vector/erase2
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *  \tparam Vector must be a Thrust vector type.
 *  \tparam Predicate must be a unary functor that returns \c true for elements to erase.
 *
 *  The following code snippet demonstrates how to use \p erase_if
 *  to remove all elements satisfying a predicate from a vector using the \p thrust::device parallelization policy:
 *
 *  \code
 *  #include <thrust/erase.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/execution_policy.h>
 *  ...
 *
 *  struct is_negative
 *  {
 *    __host__ __device__
 *    bool operator()(int x) { return x < 0; }
 *  };
 *
 *  thrust::device_vector<int> vec{1, -2, 3, -4, 5};
 *  ...
 *
 *  auto erased = thrust::erase_if(thrust::device, vec, is_negative());
 *
 *  // vec is now {1, 3, 5}
 *  // erased is 2
 *  \endcode
 *
 *  \verbatim embed:rst:leading-asterisk
 *     .. versionadded:: ?.?.?
 *  \endverbatim
 */
template <typename DerivedPolicy,
          class Vector,
          class Predicate,
          ::cuda::std::enable_if_t<is_thrust_vector_v<Vector>, int> = 0>
_CCCL_HOST typename Vector::size_type
erase_if(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Vector& c, Predicate pred);

THRUST_NAMESPACE_END

#include <thrust/detail/erase.inl>
