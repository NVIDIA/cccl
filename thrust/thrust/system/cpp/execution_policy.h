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

/*! \file thrust/system/cpp/execution_policy.h
 *  \brief Execution policies for Thrust's Standard C++ system.
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

// get the execution policies definitions first
#include <thrust/system/cpp/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/cpp/detail/par.h>

// define these entities here for the purpose of Doxygenating them
// they are actually defined elsewhere
#if _CCCL_DOXYGEN_INVOKED
THRUST_NAMESPACE_BEGIN
namespace system
{
namespace cpp
{

/*! \addtogroup execution_policies
 *  \{
 */

/*! \p thrust::system::cpp::execution_policy is the base class for all Thrust parallel execution
 *  policies which are derived from Thrust's standard C++ backend system.
 */
template <typename DerivedPolicy>
struct execution_policy : thrust::execution_policy<DerivedPolicy>
{};

/*! \p thrust::system::cpp::tag is a type representing Thrust's standard C++ backend system in C++'s type system.
 *  Iterators "tagged" with a type which is convertible to \p cpp::tag assert that they may be
 *  "dispatched" to algorithm implementations in the \p cpp system.
 */
struct tag : thrust::system::cpp::execution_policy<tag>
{
  unspecified
};

/*!
 *  \p thrust::system::cpp::par is the parallel execution policy associated with Thrust's standard
 *  C++ backend system.
 *
 *  Instead of relying on implicit algorithm dispatch through iterator system tags, users may
 *  directly target Thrust's C++ backend system by providing \p thrust::cpp::par as an algorithm
 *  parameter.
 *
 *  Explicit dispatch can be useful in avoiding the introduction of data copies into containers such
 *  as \p thrust::cpp::vector.
 *
 *  The type of \p thrust::cpp::par is implementation-defined.
 *
 *  The following code snippet demonstrates how to use \p thrust::cpp::par to explicitly dispatch an
 *  invocation of \p thrust::for_each to the standard C++ backend system:
 *
 *  \code
 *  #include <thrust/for_each.h>
 *  #include <thrust/system/cpp/execution_policy.h>
 *  #include <cstdio>
 *
 *  struct printf_functor
 *  {
 *    __host__ __device__
 *    void operator()(int x)
 *    {
 *      printf("%d\n", x);
 *    }
 *  };
 *  ...
 *  int vec[3];
 *  vec[0] = 0; vec[1] = 1; vec[2] = 2;
 *
 *  thrust::for_each(thrust::cpp::par, vec.begin(), vec.end(), printf_functor());
 *
 *  // 0 1 2 is printed to standard output in some unspecified order
 *  \endcode
 */
static const unspecified par;

/*! \}
 */

} // namespace cpp
} // namespace system
THRUST_NAMESPACE_END
#endif
