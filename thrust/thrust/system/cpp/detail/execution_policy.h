// SPDX-FileCopyrightText: Copyright (c) 2008-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace cpp
{
//! \addtogroup execution_policies
//! \{

//! \p thrust::cpp::tag is a type representing Thrust's standard C++ backend system in C++'s type system.
//! Iterators "tagged" with a type which is convertible to \p cpp::tag assert that they may be
//! "dispatched" to algorithm implementations in the \p cpp system.
struct tag;

//! \p thrust::cpp::execution_policy is the base class for all Thrust parallel execution
//! policies which are derived from Thrust's standard C++ backend system.
template <typename DerivedPolicy>
struct execution_policy;

template <>
struct execution_policy<tag> : system::detail::sequential::execution_policy<tag>
{};

struct tag : execution_policy<tag>
{};

// allow conversion to tag when it is not a successor
template <typename Derived>
struct execution_policy : system::detail::sequential::execution_policy<Derived>
{
  //! Deprecated [Since 3.2]
  using tag_type [[deprecated]] = tag;

  _CCCL_HOST_DEVICE operator tag() const
  {
    return {};
  }
};

namespace detail
{
struct par_t
    : execution_policy<par_t>
    , thrust::detail::allocator_aware_execution_policy<execution_policy>
{};
} // namespace detail

//! \p thrust::system::cpp::par is the parallel execution policy associated with Thrust's standard C++ backend system.
//!
//! Instead of relying on implicit algorithm dispatch through iterator system tags, users may directly target Thrust's
//! C++ backend system by providing \p thrust::cpp::par as an algorithm parameter.
//!
//! Explicit dispatch can be useful in avoiding the introduction of data copies into containers such as \p
//! thrust::cpp::vector.
//!
//! The type of \p thrust::cpp::par is implementation-defined.
//!
//! The following code snippet demonstrates how to use \p thrust::cpp::par to explicitly dispatch an invocation of \p
//! thrust::for_each to the standard C++ backend system:
//!
//! \code
//! #include <thrust/for_each.h>
//! #include <thrust/system/cpp/execution_policy.h>
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
//! thrust::for_each(thrust::cpp::par, vec.begin(), vec.end(), printf_functor{});
//!
//! // 0 1 2 is printed to standard output in some unspecified order
//! \endcode
_CCCL_GLOBAL_CONSTANT detail::par_t par;

//! \}

} // namespace cpp

// aliases into the system namespace

namespace system::cpp::detail // TODO(bgruber): can we drop the aliases into the detail namespace?
{
using thrust::cpp::execution_policy;
using thrust::cpp::tag;
} // namespace system::cpp::detail
namespace system::cpp
{
using thrust::cpp::execution_policy;
using thrust::cpp::par;
using thrust::cpp::tag;
} // namespace system::cpp
THRUST_NAMESPACE_END
