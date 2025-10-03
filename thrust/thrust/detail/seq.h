// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
struct seq_t
    : system::detail::sequential::execution_policy<seq_t>
    , allocator_aware_execution_policy<system::detail::sequential::execution_policy>
{
  constexpr seq_t() = default;

  // allow any execution_policy to convert to the sequential one. required for minimum_system to pick it
  template <typename DerivedPolicy>
  _CCCL_HOST_DEVICE seq_t(const thrust::execution_policy<DerivedPolicy>&)
  {}
};
} // namespace detail

//! \p thrust::seq is an execution policy which requires an algorithm invocation to execute sequentially in the current
//! thread. It can not be configured by a compile-time macro.
//!
//! The type of \p thrust::seq is implementation-defined.
//!
//! The following code snippet demonstrates how to use \p thrust::seq to explicitly execute an invocation of \p
//! thrust::for_each sequentially:
//!
//! \code
//! #include <thrust/for_each.h>
//! #include <thrust/execution_policy.h>
//! #include <vector>
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
//! std::vector<int> vec{0, 1, 2};
//! thrust::for_each(thrust::seq, vec.begin(), vec.end(), printf_functor());
//!
//! // 0 1 2 is printed to standard output in sequential order
//! \endcode
//!
//! \see thrust::host
//! \see thrust::device
_CCCL_GLOBAL_CONSTANT detail::seq_t seq;

THRUST_NAMESPACE_END
