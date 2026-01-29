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
namespace system
{
namespace hpx
{
namespace detail
{

template <typename Executor, typename Parameters>
struct sequenced_policy_shim : basic_execution_policy<sequenced_policy_shim, Executor, Parameters>
{
  using sequenced_policy_shim::basic_execution_policy::basic_execution_policy;
};

using seq_t = sequenced_policy_shim<::hpx::execution::sequenced_executor,
                                    ::hpx::traits::executor_parameters_type_t<::hpx::execution::sequenced_executor>>;

} // namespace detail

_CCCL_GLOBAL_CONSTANT detail::seq_t seq;

} // namespace hpx
} // namespace system

// alias seq here
namespace hpx
{

using thrust::system::hpx::seq;

} // namespace hpx
THRUST_NAMESPACE_END

namespace hpx
{
namespace detail
{

template <typename Executor, typename Parameters>
struct is_rebound_execution_policy<THRUST_NS_QUALIFIER::system::hpx::detail::sequenced_policy_shim<Executor, Parameters>>
    : std::true_type
{};

template <typename Executor, typename Parameters>
struct is_execution_policy<THRUST_NS_QUALIFIER::system::hpx::detail::sequenced_policy_shim<Executor, Parameters>>
    : std::true_type
{};

template <typename Executor, typename Parameters>
struct is_sequenced_execution_policy<
  THRUST_NS_QUALIFIER::system::hpx::detail::sequenced_policy_shim<Executor, Parameters>> : std::true_type
{};

} // namespace detail
} // namespace hpx
