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
#include <cuda/std/__functional/compose.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
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
  _CCCL_NODEBUG_API static constexpr auto apply_sender(_Tag, _Sndr&& __sndr, _Args&&... __args) noexcept(noexcept(
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
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto transform_sender(_Sndr&& __sndr, const _Env& __env) noexcept(
    noexcept(tag_of_t<_Sndr>{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env)))
    -> __transform_sender_result_t<tag_of_t<_Sndr>, _Sndr, _Env>
  {
    return tag_of_t<_Sndr>{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env);
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr>
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto
  transform_sender(_Sndr&& __sndr) noexcept(__nothrow_movable<_Sndr>) -> _Sndr
  {
    // FUTURE TODO: add a transform for the split sender once we have a split sender
    return static_cast<_Sndr&&>(__sndr);
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr>
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto
  transform_sender(_Sndr&& __sndr, ::cuda::std::__ignore_t) noexcept(__nothrow_movable<_Sndr>) -> _Sndr
  {
    return static_cast<_Sndr&&>(__sndr);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// get_domain has the following semantics:
//
// * When used to query a receiver's environment, returns the "current" domain, which is
//   where `start` will be called on the operation state that results from connecting the
//   receiver to a sender.
// * When used to query a sender's attributes, returns the domain on which the sender's
//   operation will complete, if the sender knows.
// * When used to query a scheduler `sch`, it is equivalent to
//   `get_domain(get_env(schedule(sch)))`.
struct get_domain_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Env>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env&) const noexcept
    -> decay_t<__query_result_t<_Env, get_domain_t>>
  {
    return {};
  }

  _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT get_domain_t get_domain{};

// Used by the schedule_from and continues_on senders
struct get_domain_override_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Env>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env&) const noexcept
    -> decay_t<__query_result_t<_Env, get_domain_override_t>>
  {
    return {};
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return false;
  }
};

_CCCL_GLOBAL_CONSTANT get_domain_override_t get_domain_override{};

namespace __detail
{
// Returns the type of the first expression that is well-formed:
// - get_domain(env)
// - get_domain(_GetScheduler{}(env))
// - _Default{}
template <class _Env, class _GetScheduler, class _Default = default_domain>
using __domain_of_t = decay_t<__call_result_t<
  __first_callable<get_domain_t, ::cuda::std::__compose_t<get_domain_t, _GetScheduler>, __always<_Default>>,
  _Env>>;

template <class _Sndr, class _Default = default_domain>
_CCCL_NODEBUG_API constexpr auto __get_domain_early() noexcept
{
  return __domain_of_t<env_of_t<_Sndr>, get_completion_scheduler_t<set_value_t>, _Default>{};
}

template <class _Sndr, class _Env, class _Default = default_domain>
_CCCL_NODEBUG_API constexpr auto __get_domain_late() noexcept
{
  // Check if the sender's attributes has a get_domain_override query. If so, use that.
  // Otherwise, we fall back to using the domain from the receiver's environment.
  if constexpr (__queryable_with<env_of_t<_Sndr>, get_domain_override_t>)
  {
    return decay_t<__query_result_t<env_of_t<_Sndr>, get_domain_override_t>>{};
  }
  else
  {
    return __domain_of_t<_Env, get_scheduler_t, _Default>{};
  }
}
} // namespace __detail

template <class... _Ts>
using __early_domain_of_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__get_domain_early<_Ts...>());

template <class... _Ts>
using __late_domain_of_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__get_domain_late<_Ts...>());

template <class _Sndr, class... _Env>
using __domain_of_t _CCCL_NODEBUG_ALIAS =
  ::cuda::std::__type_call<::cuda::std::_If<sizeof...(_Env) == 0,
                                            ::cuda::std::__type_quote<__early_domain_of_t>,
                                            ::cuda::std::__type_quote<__late_domain_of_t>>,
                           _Sndr,
                           _Env...>;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_DOMAIN
