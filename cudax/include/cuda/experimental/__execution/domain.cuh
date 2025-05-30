//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_DOMAIN
#define __CUDAX_EXECUTION_DOMAIN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__execution/env.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/fwd.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// NOLINTBEGIN(misc-unused-using-decls)
using _CUDA_STD_EXEC::__query_result_t;
using _CUDA_STD_EXEC::__queryable_with;
using _CUDA_STD_EXEC::env_of_t;
// NOLINTEND(misc-unused-using-decls)

template <class _DomainOrTag, class... _Args>
using __apply_sender_result_t _CCCL_NODEBUG_ALIAS = decltype(_DomainOrTag{}.apply_sender(declval<_Args>()...));

template <class _DomainOrTag, class _Sndr, class... _Env>
using __transform_sender_result_t =
  decltype(_DomainOrTag{}.transform_sender(declval<_Sndr>(), declval<const _Env&>()...));

//! @brief A structure that selects the default set of algorithm implementations for
//! senders.
//!
//! This structure defines static member functions to handle operations on senders, such
//! as applying and transforming them. It is designed to work with CUDA's experimental
//! execution framework.
//! @see https://eel.is/c++draft/exec.domain.default
struct _CCCL_TYPE_VISIBILITY_DEFAULT default_domain
{
  //! @brief Applies a sender operation using the specified tag and arguments.
  //!
  //! @tparam _Tag The tag type that defines the operation to be applied.
  //! @tparam _Sndr The type of the sender.
  //! @tparam _Args Variadic template for additional arguments.
  //! @param _Tag The tag instance specifying the operation.
  //! @param __sndr The sender to which the operation is applied.
  //! @param __args Additional arguments for the operation.
  //! @return The result of applying the sender operation.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tag, class _Sndr, class... _Args>
  _CCCL_TRIVIAL_API static constexpr auto apply_sender(_Tag, _Sndr&& __sndr, _Args&&... __args) noexcept(noexcept(
    _Tag{}.apply_sender(declval<_Sndr>(), declval<_Args>()...))) -> __apply_sender_result_t<_Tag, _Sndr, _Args...>
  {
    return _Tag{}.apply_sender(static_cast<_Sndr&&>(__sndr), static_cast<_Args&&>(__args)...);
  }

  //! @brief Transforms a sender with an optional environment.
  //!
  //! @tparam _Sndr The type of the sender.
  //! @tparam _Env The type of the environment.
  //! @param __sndr The sender to be transformed.
  //! @param __env The environment used for the transformation.
  //! @return The result of transforming the sender with the given environment.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr, class _Env>
  _CCCL_TRIVIAL_API static constexpr auto transform_sender(_Sndr&& __sndr, const _Env& __env) noexcept(
    noexcept(tag_of_t<_Sndr>{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env)))
    -> __transform_sender_result_t<tag_of_t<_Sndr>, _Sndr, _Env>
  {
    return tag_of_t<_Sndr>{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env);
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr>
  _CCCL_TRIVIAL_API static constexpr auto transform_sender(_Sndr&& __sndr) noexcept(__nothrow_movable<_Sndr>) -> _Sndr
  {
    // FUTURE TODO: add a transform for the split sender once we have a split sender
    return static_cast<_Sndr&&>(__sndr);
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr>
  _CCCL_TRIVIAL_API static constexpr auto
  transform_sender(_Sndr&& __sndr, _CUDA_VSTD::__ignore_t) noexcept(__nothrow_movable<_Sndr>) -> _Sndr
  {
    return static_cast<_Sndr&&>(__sndr);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// get_domain
template <class _Tag>
extern _CUDA_VSTD::__undefined<_Tag> get_domain;

// Used to query a sender's attributes for the domain on which `start` will be called,
// if it knows. Also used to query a receiver's environment for the "current" domain,
// if it knows.
template <>
struct get_domain_t<start_t>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Env>
  [[nodiscard]] _CCCL_API constexpr auto operator()([[maybe_unused]] const _Env& __env) const noexcept
  {
    if constexpr (__queryable_with<_Env, get_domain_t<start_t>>)
    {
      static_assert(noexcept(__env.query(*this)));
      return __query_result_t<_Env, get_domain_t<start_t>>{};
    }
    else
    {
      return default_domain{};
    }
  }
};

// Explicitly instantiate this because of variable template weirdness in device code
template <>
_CCCL_GLOBAL_CONSTANT get_domain_t<start_t> get_domain<start_t>{};

// For querying a sender's attributes for the domain on which `set_value` will be called,
// if it knows.
template <>
struct get_domain_t<set_value_t>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Env>
  [[nodiscard]] _CCCL_API constexpr auto operator()([[maybe_unused]] const _Env& __env) const noexcept
  {
    if constexpr (__queryable_with<_Env, get_domain_t<set_value_t>>)
    {
      static_assert(noexcept(__env.query(*this)));
      return __query_result_t<_Env, get_domain_t<set_value_t>>{};
    }
    else
    {
      // As a default, senders complete on the domain they start on
      return get_domain<start_t>(__env);
    }
  }
};

// Explicitly instantiate this because of variable template weirdness in device code
template <>
_CCCL_GLOBAL_CONSTANT get_domain_t<set_value_t> get_domain<set_value_t>{};

namespace __detail
{
template <class _Env, class _GetScheduler, class _Tag>
_CCCL_TRIVIAL_API _CCCL_CONSTEVAL auto __get_domain_impl() noexcept
{
  if constexpr (__queryable_with<_Env, get_domain_t<_Tag>>)
  {
    return __query_result_t<_Env, get_domain_t<_Tag>>{};
  }
  else if constexpr (__queryable_with<_Env, _GetScheduler>)
  {
    if constexpr (__queryable_with<__query_result_t<_Env, _GetScheduler>, get_domain_t<_Tag>>)
    {
      return __query_result_t<__query_result_t<_Env, _GetScheduler>, get_domain_t<_Tag>>{};
    }
    else
    {
      return default_domain{};
    }
  }
  else
  {
    return default_domain{};
  }
  _CCCL_UNREACHABLE();
}

template <class _Sndr>
_CCCL_TRIVIAL_API constexpr auto __get_domain_early() noexcept
{
  static_assert(!__sender_for<_Sndr, schedule_from_t>);
  return __detail::__get_domain_impl<env_of_t<_Sndr>, get_completion_scheduler_t<set_value_t>, set_value_t>();
}

template <class _Sndr, class _Env>
_CCCL_TRIVIAL_API constexpr auto __get_domain_late() noexcept
{
  if constexpr (__sender_for<_Sndr, schedule_from_t>)
  {
    // schedule_from always dispatches based on the domain of the scheduler
    return __query_result_t<env_of_t<_Sndr>, get_domain_t<set_value_t>>{};
  }
  else if constexpr (__queryable_with<env_of_t<_Sndr>, get_domain_t<start_t>>)
  {
    return __query_result_t<env_of_t<_Sndr>, get_domain_t<start_t>>{};
  }
  else
  {
    return __detail::__get_domain_impl<_Env, get_scheduler_t, start_t>();
  }
}

} // namespace __detail

template <class... _Ts>
using __early_domain_of_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__get_domain_early<_Ts...>());

template <class... _Ts>
using __late_domain_of_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__get_domain_late<_Ts...>());

template <class _Sndr, class... _Env>
using __domain_of_t _CCCL_NODEBUG_ALIAS =
  _CUDA_VSTD::__type_call<_CUDA_VSTD::_If<sizeof...(_Env) == 0,
                                          _CUDA_VSTD::__type_quote<__early_domain_of_t>,
                                          _CUDA_VSTD::__type_quote<__late_domain_of_t>>,
                          _Sndr,
                          _Env...>;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_DOMAIN
