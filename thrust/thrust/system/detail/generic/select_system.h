
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
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/system/detail/generic/select_system_exists.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
namespace select_system_detail
{

// min_system case 1: both systems have the same type, just return the first one
template <typename System>
_CCCL_HOST_DEVICE System& min_system(thrust::execution_policy<System>& system1, thrust::execution_policy<System>&)
{
  return thrust::detail::derived_cast(system1);
}

// min_system case 2: systems have differing type and the first type is considered the minimum
template <typename System1, typename System2>
_CCCL_HOST_DEVICE ::cuda::std::
  enable_if_t<::cuda::std::is_same_v<System1, thrust::detail::minimum_system<System1, System2>>, System1&>
  min_system(thrust::execution_policy<System1>& system1, thrust::execution_policy<System2>&)
{
  return thrust::detail::derived_cast(system1);
}

// min_system case 3: systems have differing type and the second type is considered the minimum
template <typename System1, typename System2>
_CCCL_HOST_DEVICE ::cuda::std::
  enable_if_t<::cuda::std::is_same_v<System2, thrust::detail::minimum_system<System1, System2>>, System2&>
  min_system(thrust::execution_policy<System1>&, thrust::execution_policy<System2>& system2)
{
  return thrust::detail::derived_cast(system2);
}
} // namespace select_system_detail

template <typename Tag>
struct select_system1_exists;

template <typename Tag1, typename Tag2>
struct select_system2_exists;

template <typename Tag1, typename Tag2, typename Tag3>
struct select_system3_exists;

template <typename Tag1, typename Tag2, typename Tag3, typename Tag4>
struct select_system4_exists;

template <typename Tag1, typename Tag2, typename Tag3, typename Tag4, typename Tag5>
struct select_system5_exists;

template <typename Tag1, typename Tag2, typename Tag3, typename Tag4, typename Tag5, typename Tag6>
struct select_system6_exists;

template <typename System>
_CCCL_HOST_DEVICE typename thrust::detail::disable_if<select_system1_exists<System>::value, System&>::type
select_system(thrust::execution_policy<System>& system)
{
  return thrust::detail::derived_cast(system);
}

template <typename System1, typename System2>
_CCCL_HOST_DEVICE thrust::detail::minimum_system<System1, System2>&
select_system(thrust::execution_policy<System1>& system1, thrust::execution_policy<System2>& system2)
{
  return select_system_detail::min_system(system1, system2);
}

template <typename System1, typename System2, typename System3>
_CCCL_HOST_DEVICE typename thrust::detail::lazy_disable_if<
  select_system3_exists<System1, System2, System3>::value,
  ::cuda::std::__type_defer<thrust::detail::minimum_system<System1, System2, System3>>>::type&
select_system(thrust::execution_policy<System1>& system1,
              thrust::execution_policy<System2>& system2,
              thrust::execution_policy<System3>& system3)
{
  return select_system(select_system(system1, system2), system3);
}

template <typename System1, typename System2, typename System3, typename System4>
_CCCL_HOST_DEVICE typename thrust::detail::lazy_disable_if<
  select_system4_exists<System1, System2, System3, System4>::value,
  ::cuda::std::__type_defer<thrust::detail::minimum_system<System1, System2, System3, System4>>>::type&
select_system(thrust::execution_policy<System1>& system1,
              thrust::execution_policy<System2>& system2,
              thrust::execution_policy<System3>& system3,
              thrust::execution_policy<System4>& system4)
{
  return select_system(select_system(system1, system2, system3), system4);
}

template <typename System1, typename System2, typename System3, typename System4, typename System5>
_CCCL_HOST_DEVICE typename thrust::detail::lazy_disable_if<
  select_system5_exists<System1, System2, System3, System4, System5>::value,
  ::cuda::std::__type_defer<thrust::detail::minimum_system<System1, System2, System3, System4, System5>>>::type&
select_system(thrust::execution_policy<System1>& system1,
              thrust::execution_policy<System2>& system2,
              thrust::execution_policy<System3>& system3,
              thrust::execution_policy<System4>& system4,
              thrust::execution_policy<System5>& system5)
{
  return select_system(select_system(system1, system2, system3, system4), system5);
}

template <typename System1, typename System2, typename System3, typename System4, typename System5, typename System6>
_CCCL_HOST_DEVICE typename thrust::detail::lazy_disable_if<
  select_system6_exists<System1, System2, System3, System4, System5, System6>::value,
  ::cuda::std::__type_defer<thrust::detail::minimum_system<System1, System2, System3, System4, System5, System6>>>::type&
select_system(thrust::execution_policy<System1>& system1,
              thrust::execution_policy<System2>& system2,
              thrust::execution_policy<System3>& system3,
              thrust::execution_policy<System4>& system4,
              thrust::execution_policy<System5>& system5,
              thrust::execution_policy<System6>& system6)
{
  return select_system(select_system(system1, system2, system3, system4, system5), system6);
}

// Map a single any_system_tag to device_system_tag.
inline _CCCL_HOST_DEVICE thrust::device_system_tag select_system(thrust::any_system_tag)
{
  return thrust::device_system_tag();
}

} // namespace system::detail::generic
THRUST_NAMESPACE_END
