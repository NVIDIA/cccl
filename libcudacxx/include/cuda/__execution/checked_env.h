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
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
template <class _Env, class _Original, class _Query, class... _Args>
_CCCL_API auto __checked_property_key(::cuda::execution::property_query<_Query, _Args...>) -> ::cuda::std::_If<
  ::cuda::std::execution::__queryable_with<const ::cuda::std::remove_reference_t<_Env>&, _Query, _Args...>,
  ::cuda::std::__type_list<_Original>,
  ::cuda::std::__type_list<>>;

//! @brief Environment adaptor advertising the supplied customizations selected by explicit query checks.
//!
//! @tparam _Env The wrapped environment type, stored by value or reference.
//! @tparam _Queries The query expressions checked against the adaptor.
template <class _Env, class... _Queries>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __checked_env
{
  //! @brief The checked expressions for which the wrapped environment supplies a member customization.
  using property_keys = ::cuda::std::__type_apply_q<
    ::cuda::execution::property_key_list,
    ::cuda::std::__type_concat<decltype(::cuda::__detail::__checked_property_key<_Env, _Queries>(
      ::cuda::std::execution::__detail::__as_property_query_t<_Queries>{}))...>>;

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
    __query_env_t __env = __env_;
    return __env.query(__query, ::cuda::std::forward<_Args>(__args)...);
  }

  _Env __env_;
};

template <class _Source, class _Result, class _Query, class... _Args>
_CCCL_API constexpr void __validate_checked_query(::cuda::execution::property_query<_Query, _Args...> __query) noexcept
{
  // Check the original advertisement before replacing the list, including entries filtering would omit.
  constexpr bool __source_is_valid =
    !::cuda::std::execution::__detail::__advertises_query_v<_Source, _Query, _Args...>
    || ::cuda::std::execution::__queryable_with<const ::cuda::std::remove_reference_t<_Source>&, _Query, _Args...>;
  static_assert(__source_is_valid,
                "The environment advertises a query expression that is not valid for a const environment.");
  if constexpr (__source_is_valid)
  {
    constexpr bool __callable = ::cuda::std::__is_callable_v<_Query, const _Result&, _Args...>;
    static_assert(__callable, "The requested query expression is not valid for a const environment.");
    if constexpr (__callable)
    {
      ::cuda::std::execution::__detail::__validate_property_query_body<_Result>(__query);
    }
  }
}
} // namespace __detail

//! @brief Explicitly validates an environment against a specified set of query expressions.
//!
//! A bare query key in @c _Queries is shorthand for
//! @c cuda::execution::property_query<Query>, a query with no additional arguments. Each expression
//! explicitly named in @c _Queries must support invocation of its query customization point object on a
//! const view of the adaptor with the specified arguments. The query may use its default answer when no
//! member customization is supplied. If the wrapped environment advertises a requested customization,
//! that member must be present. Validation does not execute the queries.
//! The returned adaptor advertises the checked expressions for which the wrapped environment supplies
//! a member customization. Expressions that use only a default answer are checked without being advertised.
//! The advertised list does not restrict forwarding: the adaptor also accepts other query expressions
//! supported by the wrapped environment's const view.
//!
//! Because the adaptor supplies its own advertised query list, it can also add metadata to a type that
//! cannot be modified or replace the query list advertised by the wrapped environment.
//!
//! The environment is stored by value by default. Pass a @c cuda::std::reference_wrapper to store
//! it by reference.
//!
//! @tparam _Queries The query expressions to explicitly validate.
//! @tparam _Env The type of the environment argument.
//! @param[in] __env The environment to wrap.
//! @return An adaptor that forwards queries and advertises the supplied customizations that were checked.
template <class... _Queries, class _Env>
[[nodiscard]] _CCCL_NODEBUG_API constexpr auto checked_env(_Env&& __env)
  -> __detail::__checked_env<::cuda::std::unwrap_ref_decay_t<_Env>, _Queries...>
{
  using __source_t _CCCL_NODEBUG = ::cuda::std::unwrap_ref_decay_t<_Env>;
  using __result_t _CCCL_NODEBUG = __detail::__checked_env<__source_t, _Queries...>;
  __result_t __result{::cuda::std::forward<_Env>(__env)};
  (::cuda::__detail::__validate_checked_query<__source_t, __result_t>(
     ::cuda::std::execution::__detail::__as_property_query_t<_Queries>{}),
   ...);
  return __result;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___EXECUTION_CHECKED_ENV_H
