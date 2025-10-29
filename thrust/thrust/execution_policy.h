// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! \file
//! \brief Thrust execution policies.

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
#include <thrust/detail/seq.h>

//! \cond

#include __THRUST_HOST_SYSTEM_ALGORITH_HEADER_INCLUDE(execution_policy.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_HEADER_INCLUDE(execution_policy.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/execution_policy.h>
#  include <thrust/system/cuda/execution_policy.h>
#  include <thrust/system/omp/execution_policy.h>
#  include <thrust/system/tbb/execution_policy.h>
#endif

//! \endcond

THRUST_NAMESPACE_BEGIN

//! \cond
namespace detail
{
using host_t   = thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::detail::par_t;
using device_t = thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::detail::par_t;
} // namespace detail
//! \endcond

//! \addtogroup execution_policies Parallel Execution Policies
//! \{

//! \p host_execution_policy is the base class for all Thrust parallel execution policies which are derived from
//! Thrust's default host backend system configured with the \p THRUST_HOST_SYSTEM macro.
//!
//! Custom user-defined backends which wish to inherit the functionality of Thrust's host backend system should derive a
//! policy from this type in order to interoperate with Thrust algorithm dispatch.
//!
//! The following code snippet demonstrates how to derive a standalone custom execution policy from \p
//! thrust::host_execution_policy to implement a backend which specializes \p for_each while inheriting the behavior of
//! every other algorithm from the host system:
//!
//! \code
//! #include <thrust/execution_policy.h>
//! #include <iostream>
//!
//! // define a type derived from thrust::host_execution_policy to distinguish our custom execution policy:
//! struct my_policy : thrust::host_execution_policy<my_policy> {};
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
//!   // dispatch thrust::transform whose behavior our policy inherits
//!   thrust::transform(exec, data, data, + 4, data, ::cuda::std::identity{});
//!
//!   return 0;
//! }
//! \endcode
//!
//! \see execution_policy
//! \see device_execution_policy
template <typename DerivedPolicy>
struct host_execution_policy : thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::execution_policy<DerivedPolicy>
{};

//! \p device_execution_policy is the base class for all Thrust parallel execution policies which are derived from
//! Thrust's default device backend system configured with the \p THRUST_DEVICE_SYSTEM macro.
//!
//! Custom user-defined backends which wish to inherit the functionality of Thrust's device backend system should derive
//! a policy from this type in order to interoperate with Thrust algorithm dispatch.
//!
//! The following code snippet demonstrates how to derive a standalone custom execution policy from \p
//! thrust::device_execution_policy to implement a backend which specializes \p for_each while inheriting the behavior
//! of every other algorithm from the device system:
//!
//! \code
//! #include <thrust/execution_policy.h>
//! #include <iostream>
//!
//! // define a type derived from thrust::device_execution_policy to distinguish our custom execution policy:
//! struct my_policy : thrust::device_execution_policy<my_policy> {};
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
//!   // dispatch thrust::transform whose behavior our policy inherits
//!   thrust::transform(exec, data, data, + 4, data, ::cuda::std::identity{});
//!
//!   return 0;
//! }
//! \endcode
//!
//! \see execution_policy
//! \see host_execution_policy
template <typename DerivedPolicy>
struct device_execution_policy : thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::execution_policy<DerivedPolicy>
{};

//! \p thrust::host is the default parallel execution policy associated with Thrust's host backend system configured by
//! the \p THRUST_HOST_SYSTEM macro.
//!
//! Instead of relying on implicit algorithm dispatch through iterator system tags, users may directly target algorithm
//! dispatch at Thrust's host system by providing \p thrust::host as an algorithm parameter.
//!
//! Explicit dispatch can be useful in avoiding the introduction of data copies into containers such as \p
//! thrust::host_vector.
//!
//! Note that even though \p thrust::host targets the host CPU, it is a parallel execution policy. That is, the order
//! that an algorithm invokes functors or dereferences iterators is not defined.
//!
//! The type of \p thrust::host is implementation-defined.
//!
//! The following code snippet demonstrates how to use \p thrust::host to explicitly dispatch an invocation of \p
//! thrust::for_each to the host backend system:
//!
//! \code
//! #include <thrust/for_each.h>
//! #include <thrust/execution_policy.h>
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
//! int vec[] = { 0, 1, 2 };
//! thrust::for_each(thrust::host, vec, vec + 3, printf_functor());
//!
//! // 0 1 2 is printed to standard output in some unspecified order
//! \endcode
//!
//! \see host_execution_policy
//! \see thrust::device
inline constexpr detail::host_t host;

//! \p thrust::device is the default parallel execution policy associated with Thrust's device backend system configured
//! by the \p THRUST_DEVICE_SYSTEM macro.
//!
//! Instead of relying on implicit algorithm dispatch through iterator system tags, users may directly target algorithm
//! dispatch at Thrust's device system by providing \p thrust::device as an algorithm parameter.
//!
//! Explicit dispatch can be useful in avoiding the introduction of data copies into containers such as \p
//! thrust::device_vector or to avoid wrapping e.g. raw pointers allocated by the CUDA API with types such as \p
//! thrust::device_ptr.
//!
//! The user must take care to guarantee that the iterators provided to an algorithm are compatible with the device
//! backend system. For example, raw pointers allocated by <tt>std::malloc</tt> typically cannot be dereferenced by a
//! GPU. For this reason, raw pointers allocated by host APIs should not be mixed with a \p thrust::device algorithm
//! invocation when the device backend is CUDA.
//!
//! The type of \p thrust::device is implementation-defined.
//!
//! The following code snippet demonstrates how to use \p thrust::device to explicitly dispatch an invocation of \p
//! thrust::for_each to the device backend system:
//!
//! \code
//! #include <thrust/for_each.h>
//! #include <thrust/device_vector.h>
//! #include <thrust/execution_policy.h>
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
//! thrust::device_vector<int> vec{0, 1, 2};
//! thrust::for_each(thrust::device, vec.begin(), vec.end(), printf_functor());
//!
//! // 0 1 2 is printed to standard output in some unspecified order
//! \endcode
//!
//! \see host_execution_policy
//! \see thrust::device
_CCCL_GLOBAL_CONSTANT detail::device_t device;

//! \}

THRUST_NAMESPACE_END
