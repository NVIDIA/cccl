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
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb
{
namespace detail
{
// note: the tag and execution policy need to be defined in the same namespace as the algorithms for ADL to find them
struct tag;

template <typename>
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
} // namespace detail

//! \addtogroup execution_policies
//! \{

//! \p thrust::tbb::tag is a type representing Thrust's Threading Building Blocks (TBB) backend system in C++'s type
//! system. Iterators "tagged" with a type which is convertible to \p tbb::tag assert that they may be "dispatched" to
//! algorithm implementations in the \p tbb system.
using detail::tag;

//! \p thrust::tbb::execution_policy is the base class for all Thrust parallel execution policies which are derived from
//! Thrust's TBB backend system.
using detail::execution_policy;

//! \p thrust::tbb::par is the parallel execution policy associated with Thrust's TBB backend system.
//!
//! Instead of relying on implicit algorithm dispatch through iterator system tags, users may directly target Thrust's
//! TBB backend system by providing \p thrust::tbb::par as an algorithm parameter.
//!
//! Explicit dispatch can be useful in avoiding the introduction of data copies into containers such as \p
//! thrust::tbb::vector.
//!
//! The type of \p thrust::tbb::par is implementation-defined.
//!
//! The following code snippet demonstrates how to use \p thrust::tbb::par to explicitly dispatch an invocation of \p
//! thrust::for_each to the TBB backend system:
//!
//! \code
//! #include <thrust/for_each.h>
//! #include <thrust/system/tbb/execution_policy.h>
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
//! thrust::for_each(thrust::tbb::par, vec.begin(), vec.end(), printf_functor());
//!
//! // 0 1 2 is printed to standard output in some unspecified order
//! \endcode
inline constexpr detail::par_t par;

//! \}
} // namespace system::tbb

// aliases:
namespace tbb
{
using system::tbb::execution_policy;
using system::tbb::par;
using system::tbb::tag;
} // namespace tbb
THRUST_NAMESPACE_END
