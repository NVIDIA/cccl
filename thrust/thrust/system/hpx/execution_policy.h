/*
 *  Copyright 2008-2025 NVIDIA Corporation
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

/*! \file thrust/system/hpx/execution_policy.h
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
#include <thrust/system/hpx/detail/execution_policy.h>

// get the definition of par, par_unseq, seq, unseq
#include <thrust/system/hpx/detail/par.h>
#include <thrust/system/hpx/detail/par_unseq.h>
#include <thrust/system/hpx/detail/seq.h>
#include <thrust/system/hpx/detail/unseq.h>

// now get all the algorithm definitions

#include <thrust/system/hpx/detail/adjacent_difference.h>
#include <thrust/system/hpx/detail/assign_value.h>
#include <thrust/system/hpx/detail/binary_search.h>
#include <thrust/system/hpx/detail/copy.h>
#include <thrust/system/hpx/detail/copy_if.h>
#include <thrust/system/hpx/detail/count.h>
#include <thrust/system/hpx/detail/equal.h>
#include <thrust/system/hpx/detail/extrema.h>
#include <thrust/system/hpx/detail/fill.h>
#include <thrust/system/hpx/detail/find.h>
#include <thrust/system/hpx/detail/for_each.h>
#include <thrust/system/hpx/detail/gather.h>
#include <thrust/system/hpx/detail/generate.h>
#include <thrust/system/hpx/detail/get_value.h>
#include <thrust/system/hpx/detail/inner_product.h>
#include <thrust/system/hpx/detail/iter_swap.h>
#include <thrust/system/hpx/detail/logical.h>
#include <thrust/system/hpx/detail/malloc_and_free.h>
#include <thrust/system/hpx/detail/merge.h>
#include <thrust/system/hpx/detail/mismatch.h>
#include <thrust/system/hpx/detail/partition.h>
#include <thrust/system/hpx/detail/reduce.h>
#include <thrust/system/hpx/detail/reduce_by_key.h>
#include <thrust/system/hpx/detail/remove.h>
#include <thrust/system/hpx/detail/replace.h>
#include <thrust/system/hpx/detail/reverse.h>
#include <thrust/system/hpx/detail/scan.h>
#include <thrust/system/hpx/detail/scan_by_key.h>
#include <thrust/system/hpx/detail/scatter.h>
#include <thrust/system/hpx/detail/sequence.h>
#include <thrust/system/hpx/detail/set_operations.h>
#include <thrust/system/hpx/detail/sort.h>
#include <thrust/system/hpx/detail/swap_ranges.h>
#include <thrust/system/hpx/detail/tabulate.h>
#include <thrust/system/hpx/detail/transform.h>
#include <thrust/system/hpx/detail/transform_reduce.h>
#include <thrust/system/hpx/detail/transform_scan.h>
#include <thrust/system/hpx/detail/uninitialized_copy.h>
#include <thrust/system/hpx/detail/uninitialized_fill.h>
#include <thrust/system/hpx/detail/unique.h>
#include <thrust/system/hpx/detail/unique_by_key.h>

// define these entities here for the purpose of Doxygenating them
// they are actually defined elsewhere
#if 0
THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{


/*! \addtogroup execution_policies
 *  \{
 */


/*! \p thrust::system::hpx::execution_policy is the base class for all Thrust parallel execution
 *  policies which are derived from Thrust's standard C++ backend system.
 */
template<typename DerivedPolicy>
struct execution_policy : thrust::execution_policy<DerivedPolicy>
{};


/*! \p thrust::system::hpx::tag is a type representing Thrust's standard C++ backend system in C++'s type system.
 *  Iterators "tagged" with a type which is convertible to \p hpx::tag assert that they may be
 *  "dispatched" to algorithm implementations in the \p hpx system.
 */
struct tag : thrust::system::hpx::execution_policy<tag> { unspecified };


/*!
 *  \p thrust::system::hpx::par is the parallel execution policy associated with Thrust's standard
 *  C++ backend system.
 *
 *  Instead of relying on implicit algorithm dispatch through iterator system tags, users may
 *  directly target Thrust's C++ backend system by providing \p thrust::hpx::par as an algorithm
 *  parameter.
 *
 *  Explicit dispatch can be useful in avoiding the introduction of data copies into containers such
 *  as \p thrust::hpx::vector.
 *
 *  The type of \p thrust::hpx::par is implementation-defined.
 *
 *  The following code snippet demonstrates how to use \p thrust::hpx::par to explicitly dispatch an
 *  invocation of \p thrust::for_each to the standard C++ backend system:
 *
 *  \code
 *  #include <thrust/for_each.h>
 *  #include <thrust/system/hpx/execution_policy.h>
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
 *  thrust::for_each(thrust::hpx::par, vec.begin(), vec.end(), printf_functor());
 *
 *  // 0 1 2 is printed to standard output in some unspecified order
 *  \endcode
 */
static const unspecified par;


/*! \}
 */


} // end hpx
} // end system
THRUST_NAMESPACE_END
#endif
