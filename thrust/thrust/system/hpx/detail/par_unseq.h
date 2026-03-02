// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <thrust/system/hpx/detail/execution_policy.h>

#include <hpx/execution.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx
{
namespace detail
{
template <typename Executor, typename Parameters>
struct parallel_unsequenced_policy_shim : basic_execution_policy<parallel_unsequenced_policy_shim, Executor, Parameters>
{
  using parallel_unsequenced_policy_shim::basic_execution_policy::basic_execution_policy;
};

using par_unseq_t =
  parallel_unsequenced_policy_shim<::hpx::execution::parallel_executor,
                                   ::hpx::traits::executor_parameters_type_t<::hpx::execution::parallel_executor>>;
} // namespace detail

/*! \p thrust::system::hpx::par_unseq is the parallel unsequenced execution policy associated with
 *  Thrust's HPX backend system.
 */

_CCCL_GLOBAL_CONSTANT detail::par_unseq_t par_unseq;
} // namespace system::hpx

// alias par_unseq here
namespace hpx
{
using thrust::system::hpx::par_unseq;
} // namespace hpx
THRUST_NAMESPACE_END

namespace hpx::detail
{
template <typename Executor, typename Parameters>
struct is_rebound_execution_policy<
  THRUST_NS_QUALIFIER::system::hpx::detail::parallel_unsequenced_policy_shim<Executor, Parameters>> : std::true_type
{};

template <typename Executor, typename Parameters>
struct is_execution_policy<
  THRUST_NS_QUALIFIER::system::hpx::detail::parallel_unsequenced_policy_shim<Executor, Parameters>> : std::true_type
{};

template <typename Executor, typename Parameters>
struct is_parallel_execution_policy<
  THRUST_NS_QUALIFIER::system::hpx::detail::parallel_unsequenced_policy_shim<Executor, Parameters>> : std::true_type
{};

template <typename Executor, typename Parameters>
struct is_unsequenced_execution_policy<
  THRUST_NS_QUALIFIER::system::hpx::detail::parallel_unsequenced_policy_shim<Executor, Parameters>> : std::true_type
{};
} // namespace hpx::detail
