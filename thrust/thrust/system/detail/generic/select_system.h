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

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

THRUST_NAMESPACE_BEGIN

namespace system::detail::generic
{
template <typename, typename... Tags>
inline constexpr bool select_system_exists_impl = false;

template <typename... Tags>
inline constexpr bool
  select_system_exists_impl<::cuda::std::void_t<decltype(select_system(::cuda::std::declval<Tags&>()...))>, Tags...> =
    true;

template <typename... Tags>
inline constexpr bool select_system_exists = select_system_exists_impl<void, Tags...>;

template <typename System, ::cuda::std::enable_if_t<!select_system_exists<System>, int> = 0>
_CCCL_HOST_DEVICE auto select_system(thrust::execution_policy<System>& system) -> System&
{
  return thrust::detail::derived_cast(system);
}

// note: previous binary implementation of select_system also did not check select_system_exists<System1, System2>
template <typename System1, typename System2>
_CCCL_HOST_DEVICE auto
select_system(thrust::execution_policy<System1>& system1, thrust::execution_policy<System2>& system2)
  -> thrust::detail::minimum_system_t<System1, System2>&
{
  using min_sys = thrust::detail::minimum_system_t<System1, System2>;
  if constexpr (::cuda::std::is_same_v<System1, System2> || ::cuda::std::is_same_v<System1, min_sys>)
  {
    return thrust::detail::derived_cast(system1);
  }
  else if constexpr (::cuda::std::is_same_v<System2, min_sys>)
  {
    return thrust::detail::derived_cast(system2);
  }
  else if constexpr (thrust::detail::is_unrelated_systems<min_sys>)
  {
    static_assert(!sizeof(System1), "Cannot select a system: System1 and System2 are unrelated");
  }
  else
  {
    static_assert(!sizeof(System1), "select_system failed. Please file a bug report!");
  }
  _CCCL_UNREACHABLE();
}

template <typename System1,
          typename System2,
          typename... Systems,
          ::cuda::std::enable_if_t<!select_system_exists<System1, System2, Systems...>, int> = 0>
_CCCL_HOST_DEVICE auto select_system(
  thrust::execution_policy<System1>& system1,
  thrust::execution_policy<System2>& system2,
  thrust::execution_policy<Systems>&... systems) -> thrust::detail::minimum_system_t<System1, System2, Systems...>&
{
  return select_system(select_system(system1, system2), systems...);
}

// Map a single any_system_tag to device_system_tag.
inline _CCCL_HOST_DEVICE thrust::device_system_tag select_system(thrust::any_system_tag)
{
  return thrust::device_system_tag();
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
