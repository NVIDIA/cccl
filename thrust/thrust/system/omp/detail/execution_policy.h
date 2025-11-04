// SPDX-FileCopyrightText: Copyright (c) 2008-2025, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/allocator_aware_execution_policy.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp
{
namespace detail
{
// note: the tag and execution policy need to be defined in the same namespace as the algorithms for ADL to find them
struct tag;

template <typename Derived>
struct execution_policy;

template <>
struct execution_policy<tag> : cpp::execution_policy<tag>
{
  using tag_type = tag;
};

struct tag : execution_policy<tag>
{};

template <typename Derived>
struct execution_policy : cpp::execution_policy<Derived>
{
  using tag_type = tag;

  // allow conversion to tag when it is not a successor
  operator tag() const
  {
    return tag();
  }
};

struct par_t
    : execution_policy<par_t>
    , thrust::detail::allocator_aware_execution_policy<execution_policy>
{};

// select_system(tbb, omp) & select_system(omp, tbb) are ambiguous because both convert to cpp without these overloads,
// which we arbitrarily define in the omp backend

template <typename System1, typename System2>
_CCCL_HOST_DEVICE System1 select_system(execution_policy<System1> s, tbb::execution_policy<System2>)
{
  return thrust::detail::derived_cast(s);
}

template <typename System1, typename System2>
_CCCL_HOST_DEVICE System2 select_system(tbb::execution_policy<System1>, execution_policy<System2> s)
{
  return thrust::detail::derived_cast(s);
}
} // namespace detail

//! \addtogroup execution_policies
//! \{

//! \p thrust::omp::tag is a type representing Thrust's OpenMP backend system in C++'s type system. Iterators "tagged"
//! with a type which is convertible to \p omp::tag assert that they may be "dispatched" to algorithm implementations in
//! the \p omp system.
using detail::tag;

//! \p thrust::omp::execution_policy is the base class for all Thrust parallel execution policies which are derived from
//! Thrust's OpenMP backend system.
using detail::execution_policy;

//! \p thrust::omp::par is the parallel execution policy associated with Thrust's OpenMP backend system.
//!
//! Instead of relying on implicit algorithm dispatch through iterator system tags, users may directly target Thrust's
//! OpenMP backend system by providing \p thrust::omp::par as an algorithm parameter.
//!
//! Explicit dispatch can be useful in avoiding the introduction of data copies into containers such as \p
//! thrust::omp::vector.
//!
//! The type of \p thrust::omp::par is implementation-defined.
//!
//! The following code snippet demonstrates how to use \p thrust::omp::par to explicitly dispatch an invocation of \p
//! thrust::for_each to the OpenMP backend system:
//!
//! \code
//! #include <thrust/for_each.h>
//! #include <thrust/system/omp/execution_policy.h>
//! #include <cstdio>
//!
//! struct printf_functor
//! {
//!   __host__ __device__
//!   void operator()(int x)
//!   {
//!     printf("%d\n", x);
//!   }
//! };
//! ...
//! int vec[3]{0, 1, 2}
//! thrust::for_each(thrust::omp::par, vec.begin(), vec.end(), printf_functor{});
//!
//! // 0 1 2 is printed to standard output in some unspecified order
//! \endcode
inline constexpr detail::par_t par;

//! \}
} // namespace system::omp

// aliases:
namespace omp
{
using system::omp::execution_policy;
using system::omp::par;
using system::omp::tag;
} // namespace omp
THRUST_NAMESPACE_END
