//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_DOMAIN
#define __CUDAX_ASYNC_DETAIL_DOMAIN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__execution/env.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/fwd.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
using _CUDA_STD_EXEC::__query_result_t;
using _CUDA_STD_EXEC::__queryable_with;
using _CUDA_STD_EXEC::env_of_t;

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
  template <class _Sndr, class _Env>
  _CCCL_TRIVIAL_API static constexpr auto transform_sender(_Sndr&& __sndr, const _Env& __env) noexcept(
    noexcept(tag_of_t<_Sndr>{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env)))
    -> __transform_sender_result_t<tag_of_t<_Sndr>, _Sndr, _Env>
  {
    return tag_of_t<_Sndr>{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env);
  }

  //! @overload
  template <class _Sndr>
  _CCCL_TRIVIAL_API static constexpr auto transform_sender(_Sndr&& __sndr) noexcept -> _Sndr&&
  {
    // FUTURE TODO: add a transform for the split sender once we have a split sender
    return static_cast<_Sndr&&>(__sndr);
  }

  //! @overload
  template <class _Sndr>
  _CCCL_TRIVIAL_API static constexpr auto transform_sender(_Sndr&& __sndr, _CUDA_VSTD::__ignore_t) noexcept -> _Sndr&&
  {
    return static_cast<_Sndr&&>(__sndr);
  }
};

namespace __detail
{
// This function is used to determine the domain associated with a sender and
// (optionally), an environment. It uses clues from the sender and environment to deduce
// where the sender will execute (on what scheduler), and what the domain of that
// scheduler is. It proceeds as follows:
// 1. If the sender attributes has a domain, return it.
// 2. Otherwise, if the sender has a completion scheduler, return its domain if any.
// 3. Otherwise, if an environment is provided:
//   - If the environment has a domain, return it.
//   - If the environment has a scheduler, return its domain if any.
// 4. Otherwise, return the default domain.
template <class _GetScheduler, class _Env1, class _Env2 = _CUDA_STD_EXEC::prop<get_domain_t, default_domain>>
_CCCL_TRIVIAL_API constexpr auto __domain_for() noexcept
{
  if constexpr (__queryable_with<_Env1, get_domain_t>)
  {
    return __query_result_t<_Env1, get_domain_t>{};
  }
  else if constexpr (__queryable_with<_Env1, _GetScheduler>)
  {
    if constexpr (__queryable_with<__query_result_t<_Env1, _GetScheduler>, get_domain_t>)
    {
      return __query_result_t<__query_result_t<_Env1, _GetScheduler>, get_domain_t>{};
    }
    else
    {
      return __domain_for<get_scheduler_t, _Env2>();
    }
  }
  else
  {
    return __domain_for<get_scheduler_t, _Env2>();
  }
}
} // namespace __detail

template <class _Sndr, class... _Env>
using domain_for_t _CCCL_NODEBUG_ALIAS =
  decltype(__detail::__domain_for<get_completion_scheduler_t<set_value_t>, env_of_t<_Sndr>, _Env...>());

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_DOMAIN
