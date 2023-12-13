/*
 *  Copyright 2008-2018 NVIDIA Corporation
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
#include <thrust/detail/cpp11_required.h>

#if _CCCL_STD_VER >= 2011

#include <thrust/detail/type_deduction.h>
#include <thrust/type_traits/remove_cvref.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

// We need a way to compute the return type of `select_system`, which is found
// by using `thrust::system::detail::generic::select_system` and then making an
// ADL call. We have no trait that defines the return type. With the
// limitations of C++11 return type deduction, we need to be able to stick all
// of that into `decltype`. So, we put the using statement into a detail
// namespace, and then implement the generic dispatch function in that
// namespace.

namespace select_system_detail
{

using thrust::system::detail::generic::select_system;

struct select_system_fn final
{
  _CCCL_EXEC_CHECK_DISABLE
  template <typename DerivedPolicy0>
  _CCCL_HOST_DEVICE
  auto operator()(
    thrust::detail::execution_policy_base<DerivedPolicy0> const& exec0
  ) const
  THRUST_DECLTYPE_RETURNS(
    select_system(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec0))
    )
  )

  _CCCL_EXEC_CHECK_DISABLE
  template <typename DerivedPolicy0, typename DerivedPolicy1>
  _CCCL_HOST_DEVICE
  auto operator()(
    thrust::detail::execution_policy_base<DerivedPolicy0> const& exec0
  , thrust::detail::execution_policy_base<DerivedPolicy1> const& exec1
  ) const
  THRUST_DECLTYPE_RETURNS(
    select_system(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec0))
    , thrust::detail::derived_cast(thrust::detail::strip_const(exec1))
    )
  )
};

} // namespace select_system_detail

THRUST_INLINE_CONSTANT select_system_detail::select_system_fn select_system{};

} // detail

THRUST_NAMESPACE_END

#endif // _CCCL_STD_VER >= 2011

