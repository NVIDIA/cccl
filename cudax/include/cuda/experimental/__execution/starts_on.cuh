//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STARTS_ON
#define __CUDAX_EXECUTION_STARTS_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__execution/continues_on.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/just.cuh>
#include <cuda/experimental/__execution/sequence.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//! \brief Execution algorithm that starts a given sender on a specified scheduler.
//!
//! The `starts_on` algorithm takes a scheduler and a sender, and returns a new sender
//! that, when connected and started, will first schedule work on the provided scheduler,
//! and then start the original sender on that scheduler's execution context.
//!
//! This algorithm is particularly useful for ensuring that a chain of work begins
//! execution on a specific execution context, such as a particular GPU stream or thread
//! pool.
//!
//! \details The operation proceeds in two phases:
//! 1. **Scheduling Phase**: The algorithm first calls `schedule()` on the provided
//!    scheduler to obtain a sender that represents scheduling work on that scheduler's
//!    execution context.
//! 2. **Execution Phase**: Once the scheduling operation completes successfully, the
//!    original sender is started on the scheduler's execution context.
//!
//! The resulting sender's completion signatures are derived from both the scheduler's
//! `schedule()` sender and the original sender. Error and stopped signals from either
//! operation are propagated to the final receiver.
//!
//! \tparam _Sch A scheduler type that satisfies the `scheduler` concept
//! \tparam _Sndr A sender type that satisfies the `sender` concept
//!
//! \param __sch The scheduler on which the sender should start execution
//! \param __sndr The sender to be started on the scheduler's execution context
//!
//! \return A sender that, when started, will first schedule on `__sch` and then execute
//!         `__sndr`
//!
//! \note The returned sender's environment includes the provided scheduler as the current
//!       scheduler, allowing nested senders to query and use the same execution context.
//!
//! \note This implementation follows the C++26 standard specification for
//!       `std::execution::starts_on` as defined in [exec.starts.on].
//!
//! Example usage:
//! \code
//! auto work = cuda::experimental::execution::just(42)
//!           | cuda::experimental::execution::then([](int x) { return x * 2; });
//!
//! auto scheduled_work = cuda::experimental::execution::starts_on(some_scheduler, work);
//! \endcode
//!
//! \see schedule
//! \see scheduler
//! \see sender
//! \see receiver
struct starts_on_t
{
  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sndr>
  [[nodiscard]] static _CCCL_API constexpr auto transform_sender(_Sndr&& __sndr, ::cuda::std::__ignore_t)
  {
    auto&& [__ign, __sch, __child] = __sndr;
    return sequence(continues_on(just(), __sch), ::cuda::std::forward_like<_Sndr>(__child));
  }

  template <class _Sch, class _Sndr>
  _CCCL_NODEBUG_API constexpr auto operator()(_Sch __sch, _Sndr __sndr) const;
};

template <class _Sch, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT starts_on_t::__sndr_t
{
  using sender_concept = sender_t;
  using __domain_t     = __query_result_or_t<_Sch, get_domain_t, __nil>;

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
  {
    template <class... _Env>
    [[nodiscard]] _CCCL_API static constexpr auto query(get_completion_behavior_t, _Env&&...) noexcept
    {
      return (execution::min) (execution::get_completion_behavior<schedule_result_t<_Sch>, __fwd_env_t<_Env>...>(),
                               execution::get_completion_behavior<_Sndr, __fwd_env_t<_Env>...>());
    }

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query) const
      noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query>) -> __query_result_t<env_of_t<_Sndr>, _Query>
    {
      return execution::get_env(__sndr_).query(_Query{});
    }

    _Sndr const& __sndr_;
  };

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    using __env_t   = ::cuda::std::__type_index_c<0, _Env..., env<>>;
    using __sndr2_t = __transform_sender_result_t<starts_on_t, _Self, __env_t>;
    return execution::get_completion_signatures<__sndr2_t, _Env...>();
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return __attrs_t{__sndr_};
  }

  _CCCL_NO_UNIQUE_ADDRESS starts_on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;
};

template <class _Sch, class _Sndr>
[[nodiscard]] _CCCL_NODEBUG_API constexpr auto starts_on_t::operator()(_Sch __sch, _Sndr __sndr) const
{
  if constexpr (__queryable_with<_Sch, get_domain_t>)
  {
    using __domain_t = __query_result_t<_Sch, get_domain_t>;
    return execution::transform_sender(
      __domain_t{}, __sndr_t<_Sch, _Sndr>{{}, static_cast<_Sch&&>(__sch), static_cast<_Sndr&&>(__sndr)});
  }
  else
  {
    return __sndr_t<_Sch, _Sndr>{{}, static_cast<_Sch&&>(__sch), static_cast<_Sndr&&>(__sndr)};
  }
}

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<starts_on_t::__sndr_t<_Sch, _Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT starts_on_t starts_on{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STARTS_ON
