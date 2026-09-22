//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___EXECUTION_CHECKED_ENV_H
#define _CUDA___EXECUTION_CHECKED_ENV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/unwrap_ref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
//! @brief Environment adaptor carrying an explicit advertised query list.
//!
//! @tparam _Env The wrapped environment type, stored by value or reference.
//! @tparam _Queries The query expressions advertised by the adaptor.
template <class _Env, class... _Queries>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __checked_env
{
  //! @brief The query expressions advertised by this adaptor.
  using property_keys = ::cuda::execution::property_key_list<_Queries...>;

private:
  using __query_env_t _CCCL_NODEBUG = const ::cuda::std::remove_reference_t<_Env>&;

public:
  //! @brief Forwards a query to the wrapped environment through a const reference.
  //!
  //! @tparam _Query The query key type.
  //! @tparam _Args The additional query argument types.
  //! @param[in] __query The query key.
  //! @param[in] __args The additional query arguments.
  //! @return The result of querying the wrapped environment.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Query, class... _Args)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<__query_env_t, _Query, _Args...>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(_Query __query, _Args&&... __args) const
    noexcept(::cuda::std::execution::__nothrow_queryable_with<__query_env_t, _Query, _Args...>)
      -> ::cuda::std::execution::__query_result_t<__query_env_t, _Query, _Args...>
  {
    const ::cuda::std::remove_reference_t<_Env>& __env = __env_;
    return __env.query(__query, static_cast<_Args&&>(__args)...);
  }

  _Env __env_;
};
} // namespace __detail

//! @brief Explicitly validates an environment against a specified set of query expressions.
//!
//! A bare query key in @c _Queries is shorthand for
//! @c cuda::execution::property_query<Query>, a query with no additional arguments. Each expression
//! explicitly named in @c _Queries is checked against a const view of the wrapped environment. The
//! returned adaptor records those expressions as its advertised query list and forwards queries to the
//! wrapped environment.
//!
//! Because the adaptor supplies its own advertised query list, it can also add metadata to a type that
//! cannot be modified or replace the query list advertised by the wrapped environment.
//!
//! The environment is stored by value by default. Pass a @c cuda::std::reference_wrapper to store
//! it by reference.
//!
//! @tparam _Queries The query expressions to explicitly validate and advertise.
//! @tparam _Env The type of the environment argument.
//! @param[in] __env The environment to wrap.
//! @return An adaptor that records the validated query list and forwards queries to its stored environment.
template <class... _Queries, class _Env>
[[nodiscard]] _CCCL_NODEBUG_API constexpr auto checked_env(_Env&& __env)
  -> __detail::__checked_env<::cuda::std::unwrap_ref_decay_t<_Env>, _Queries...>
{
  using __result_t _CCCL_NODEBUG = __detail::__checked_env<::cuda::std::unwrap_ref_decay_t<_Env>, _Queries...>;
  __result_t __result{::cuda::std::forward<_Env>(__env)};
  ::cuda::std::execution::__detail::__validate_env<__result_t>();
  return __result;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___EXECUTION_CHECKED_ENV_H
