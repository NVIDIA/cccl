// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

THRUST_NAMESPACE_BEGIN

namespace detail
{
struct execution_policy_marker
{};

// execution_policy_base serves as a guard against infinite recursion in thrust entry points:
//
// template<typename DerivedPolicy>
// void foo(const thrust::detail::execution_policy_base<DerivedPolicy> &s)
// {
//   using thrust::system::detail::generic::foo;
//
//   foo(thrust::detail::derived_cast(thrust::detail::strip_const(s));
// }
//
// foo is not recursive when
// 1. DerivedPolicy is derived from thrust::execution_policy below
// 2. generic::foo takes thrust::execution_policy as a parameter
template <typename DerivedPolicy>
struct execution_policy_base : execution_policy_marker
{};

template <typename DerivedPolicy>
constexpr _CCCL_HOST_DEVICE execution_policy_base<DerivedPolicy>&
strip_const(const execution_policy_base<DerivedPolicy>& x)
{
  return const_cast<execution_policy_base<DerivedPolicy>&>(x);
}

template <typename DerivedPolicy>
constexpr _CCCL_HOST_DEVICE DerivedPolicy& derived_cast(execution_policy_base<DerivedPolicy>& x)
{
  return static_cast<DerivedPolicy&>(x);
}

template <typename DerivedPolicy>
constexpr _CCCL_HOST_DEVICE const DerivedPolicy& derived_cast(const execution_policy_base<DerivedPolicy>& x)
{
  return static_cast<const DerivedPolicy&>(x);
}
} // namespace detail

//! \addtogroup execution_policies
//! \{

//! \p execution_policy is the base class for all Thrust parallel execution policies like \p thrust::host, \p
//! thrust::device, and each backend system's tag type.
//!
//! Custom user-defined backends should derive a policy from this type in order to interoperate with Thrust algorithm
//! dispatch.
//!
//! The following code snippet demonstrates how to derive a standalone custom execution policy from \p
//! thrust::execution_policy to implement a backend which only implements \p for_each:
//!
//! \code
//! #include <thrust/execution_policy.h>
//! #include <iostream>
//!
//! // define a type derived from thrust::execution_policy to distinguish our custom execution policy:
//! struct my_policy : thrust::execution_policy<my_policy> {};
//!
//! // overload for_each on my_policy
//! template<typename Iterator, typename Function>
//! Iterator for_each(my_policy, Iterator first, Iterator last, Function f)
//! {
//!   std::cout << "Hello, world from for_each(my_policy)!" << std::endl;
//!
//!   for(; first < last; ++first)
//!   {
//!     f(*first);
//!   }
//!
//!   return first;
//! }
//!
//! struct ignore_argument
//! {
//!   void operator()(int) {}
//! };
//!
//! int main()
//! {
//!   int data[4];
//!
//!   // dispatch thrust::for_each using our custom policy:
//!   my_policy exec;
//!   thrust::for_each(exec, data, data + 4, ignore_argument());
//!
//!   // can't dispatch thrust::transform because no overload exists for my_policy:
//!   //thrust::transform(exec, data, data, + 4, data, ::cuda::std::identity{}); // error!
//!
//!   return 0;
//! }
//! \endcode
//!
//! \see host_execution_policy
//! \see device_execution_policy
template <typename DerivedPolicy>
struct execution_policy : detail::execution_policy_base<DerivedPolicy>
{};

//! \}

THRUST_NAMESPACE_END
