//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_ON
#define __CUDAX_EXECUTION_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/continues_on.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/starts_on.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/visit.cuh>
#include <cuda/experimental/__execution/write_env.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//! @brief Sender adaptor that transfers execution to a specified scheduler and back.
//!
//! The `on` algorithm provides execution context control by moving computation to
//! different execution resources. It has two primary forms:
//!
//! ## Form 1: `on(scheduler, sender)`
//!
//! Starts a sender on an execution agent belonging to the specified scheduler's execution
//! resource, and upon completion, transfers execution back to the original execution
//! resource where the `on` sender was started.
//!
//! @code
//! auto result = on(gpu_scheduler, some_computation) | sync_wait();
//! @endcode
//!
//! ## Form 2: `on(sender, scheduler, closure)` or `sender | on(scheduler, closure)`
//!
//! Upon completion of the input sender, transfers execution to the specified scheduler's
//! execution resource, executes the closure with the sender's results, and then transfers
//! execution back to where the original sender completed.
//!
//! @code
//! auto result = some_computation | on(gpu_scheduler, then([](auto value) { return process(value); }));
//! @endcode
//!
//! ## Behavior
//!
//! - **Form 1**: Execution flow: current → target scheduler → back to current
//! - **Form 2**: Execution flow: current → (sender completes) → target scheduler → back to sender's completion context
//!
//! The algorithm remembers the original scheduler context and ensures execution returns
//! to it after the target scheduler's work is complete. If no scheduler is available in
//! the current execution context, the operation is ill-formed and results in a
//! compilation error.
//!
//! ## Error Handling
//!
//! If any scheduling operation fails, an error completion is executed on an unspecified
//! execution agent.
//!
//! @note This is CUDA's experimental implementation of the C++26 `std::execution::on` algorithm
//!       as specified in [exec.on].
//!
//! @see @c starts_on and @c continues_on for related scheduling primitives
struct on_t
{
  _CUDAX_SEMI_PRIVATE :
  struct __any_t
  {
    using type = __any_t;

    template <class _Ty>
    [[nodiscard]] _CCCL_API operator _Ty&&() const noexcept;
  };

  struct __not_a_sender
  {
    using sender_concept = sender_t;

    template <class _Self, class _Env>
    [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
    {
      return invalid_completion_signature<
        _WHAT(_THE_ENVIRONMENT_OF_THE_RECEIVER_DOES_NOT_HAVE_A_SCHEDULER_FOR_ON_TO_RETURN_TO),
        _WHERE(_IN_ALGORITHM, on_t),
        _WITH_ENVIRONMENT(_Env)>();
    }
  };

  struct __not_a_scheduler
  {
    using scheduler_concept = scheduler_t;
    [[nodiscard]] _CCCL_API static constexpr auto schedule() noexcept
    {
      return __not_a_sender{};
    }
  };

  template <class _Sndr, class _NewSch, class _OldSch, class... _Closure>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __lowered_sndr_t;

  struct __lower_sndr_fn
  {
    // This is the the lowering for the `on(sch, sndr)` case
    template <class _Sndr, class _NewSch, class _OldSch>
    [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr __sndr, _NewSch __new_sch, _OldSch __old_sch) const
    {
      return continues_on(starts_on(static_cast<_NewSch&&>(__new_sch), static_cast<_Sndr&&>(__sndr)), __old_sch);
    }

    // This is the the lowering for the `sndr | on(sch, clsr)` case
    template <class _Sndr, class _NewSch, class _OldSch, class _Closure>
    [[nodiscard]] _CCCL_API constexpr auto
    operator()(_Sndr __sndr, _NewSch __new_sch, _OldSch __old_sch, _Closure __closure) const
    {
      return continues_on(static_cast<_Closure&&>(__closure)(continues_on(static_cast<_Sndr&&>(__sndr), __new_sch)),
                          __old_sch);
    }
  };

  // Helper alias for the environment of the receiver used to connect the child sender
  // in the on(sch, sndr) case.
  template <class _Sch, class _Env>
  using __env2_t = __join_env_t<__call_result_t<__mk_sch_env_t, _Sch, _Env>, _Env>;

  template <class _Sch, class _Env>
  [[nodiscard]] _CCCL_API static constexpr auto __mk_env2(_Sch __sch, _Env&& __env)
  {
    return __join_env(__mk_sch_env(__sch, __env), static_cast<_Env&&>(__env));
  }

public:
  template <class _Sch, class _Sndr, class... _Closure>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sch, class _Closure>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t
  {
    template <class _Sndr>
    [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr) &&
    {
      return on_t{}(static_cast<_Sndr&&>(__sndr), __sch_, static_cast<_Closure&&>(__closure_));
    }

    template <class _Sndr>
    [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr) const&
    {
      return on_t{}(static_cast<_Sndr&&>(__sndr), __sch_, __closure_);
    }

    template <class _Sndr>
    [[nodiscard]] _CCCL_NODEBUG_API friend constexpr auto operator|(_Sndr __sndr, __closure_t __self)
    {
      return on_t{}(static_cast<_Sndr&&>(__sndr), __self.__sch_, static_cast<_Closure&&>(__self.__closure_));
    }

    _Sch __sch_;
    _Closure __closure_;
  };

  _CCCL_TEMPLATE(class _Sch, class _Sndr)
  _CCCL_REQUIRES(__is_sender<_Sndr>)
  _CCCL_NODEBUG_API constexpr auto operator()(_Sch __sch, _Sndr __sndr) const
  {
    using __domain_t = __query_result_or_t<_Sch, get_domain_t, default_domain>;
    return execution::transform_sender(__domain_t{}, __sndr_t<_Sch, _Sndr>{{}, __sch, __sndr});
  }

  _CCCL_TEMPLATE(class _Sch, class _Closure)
  _CCCL_REQUIRES((!__is_sender<_Closure>) )
  _CCCL_NODEBUG_API constexpr auto operator()(_Sch __sch, _Closure __closure) const
  {
    return __closure_t<_Sch, _Closure>{__sch, static_cast<_Closure&&>(__closure)};
  }

  template <class _Sndr, class _Sch, class _Closure>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr, _Sch __sch, _Closure __closure) const
  {
    using __sndr_t = on_t::__sndr_t<_Sch, _Sndr, _Closure>;
    auto __domain  = __early_domain_of_t<_Sndr>{};
    return execution::transform_sender(
      __domain, __sndr_t{{}, {__sch, static_cast<_Closure&&>(__closure)}, static_cast<_Sndr&&>(__sndr)});
  }

  template <class _Sndr, class _Env>
  [[nodiscard]] _CCCL_API static constexpr auto transform_env(const _Sndr& __sndr, _Env&& __env)
  {
    auto& [__ign1, __data, __ign2] = __sndr;
    if constexpr (__is_scheduler<decltype(__data)>)
    {
      return __mk_env2(__data, static_cast<_Env&&>(__env));
    }
    else
    {
      return static_cast<_Env&&>(__env);
    }
  }

  template <class _Sndr, class _Env>
  [[nodiscard]] _CCCL_API static constexpr auto transform_sender(_Sndr&& __sndr, const _Env& __env)
  {
    auto&& [__ign, __data, __child] = __sndr;
    if constexpr (__is_scheduler<decltype(__data)>)
    {
      // The on(sch, sndr) case:
      auto __old_sch = __call_or(get_scheduler, __not_a_scheduler{}, __env);
      using __sndr_t = __lowered_sndr_t<decltype(__child), decltype(__data), decltype(__old_sch)>;
      return __sndr_t{::cuda::std::forward_like<_Sndr>(__child), __data, __old_sch};
    }
    else
    {
      // The on(sndr, sch, closure) case:
      auto& [__new_sch, __closure] = __data;
      auto __old_sch =
        __call_or(get_completion_scheduler<set_value_t>, __not_a_scheduler{}, execution::get_env(__child), __env);
      using __sndr_t =
        __lowered_sndr_t<decltype(__child), decltype(__new_sch), decltype(__old_sch), decltype(__closure)>;
      return __sndr_t{
        ::cuda::std::forward_like<_Sndr>(__child), __new_sch, __old_sch, ::cuda::std::forward_like<_Sndr>(__closure)};
    }
  }
};

template <class _Sndr, class _NewSch, class _OldSch, class... _Closure>
struct _CCCL_TYPE_VISIBILITY_DEFAULT on_t::__lowered_sndr_t
    : __call_result_t<on_t::__lower_sndr_fn, _Sndr, _NewSch, _OldSch, _Closure...>
{
  using __base_t = __call_result_t<on_t::__lower_sndr_fn, _Sndr, _NewSch, _OldSch, _Closure...>;

  _CCCL_API constexpr __lowered_sndr_t(_Sndr&& __sndr, _NewSch __new_sch, _OldSch __old_sch, _Closure... __closure)
      : __base_t{on_t::__lower_sndr_fn{}(
          static_cast<_Sndr&&>(__sndr), __new_sch, __old_sch, static_cast<_Closure&&>(__closure)...)}
  {}
};

// This is the sender used for `on(sch, sndr)`
template <class _Sch, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT on_t::__sndr_t<_Sch, _Sndr>
{
private:
  template <class _SetTag, class _Env, class _OldSch = __scheduler_of_t<_Env&>>
  [[nodiscard]] static constexpr auto __get_completion_scheduler_for(
    _SetTag, [[maybe_unused]] const __sndr_t& __self, [[maybe_unused]] _Env&& __env) noexcept
  {
    if constexpr (_SetTag{} == set_value)
    {
      // When it completes successfully, the on(sch, sndr) sender completes where it
      // starts.
      return execution::get_scheduler(__env);
    }
    else
    {
      // When an on(sch, sndr) sender is connected with rcvr and started, three senders get
      // connected and started:
      //   1. The sender returned from schedule(sch).
      //   2. sndr (which sees that the current scheduler is sch).
      //   3. The sender returned from schedule(get_scheduler(get_env(rcvr))).
      //
      // If exactly one of those senders has a _SetTag completion *and* if that sender
      // knows its completion scheduler for _SetTag, that is the completion scheduler of
      // the on(sch, sndr) sender.
      constexpr bool __new_sch_has_compl = __has_completions_for<_SetTag, schedule_result_t<_Sch>, __fwd_env_t<_Env>>;
      constexpr bool __sndr_has_compl    = __has_completions_for<_SetTag, _Sndr, __env2_t<_Sch, _Env>>;
      constexpr bool __old_sch_has_compl =
        __has_completions_for<_SetTag, schedule_result_t<_OldSch>, __fwd_env_t<_Env>>;

      if constexpr (__new_sch_has_compl + __sndr_has_compl + __old_sch_has_compl == 1)
      {
        if constexpr (__new_sch_has_compl && __callable<execution::get_completion_scheduler_t<_SetTag>, _Sch, _Env&>)
        {
          return execution::get_completion_scheduler<_SetTag>(__self.__sch_, __env);
        }
        else if constexpr (__sndr_has_compl
                           && __callable<execution::get_completion_scheduler_t<_SetTag>,
                                         env_of_t<_Sndr>,
                                         __env2_t<_Sch, _Env>>)
        {
          return execution::get_completion_scheduler<_SetTag>(
            execution::get_env(__self.__sndr_), __mk_env2(__self.__sch_, static_cast<_Env&&>(__env)));
        }
        else if constexpr (__old_sch_has_compl
                           && __callable<execution::get_completion_scheduler_t<_SetTag>, _OldSch, _Env&>)
        {
          return execution::get_completion_scheduler<_SetTag>(execution::get_scheduler(__env), __env);
        }
      }
    }
  }

  template <class _SetTag, class _Env, class _OldSch = __scheduler_of_t<_Env&>>
  [[nodiscard]] static constexpr auto
  __get_completion_domain_for(_SetTag, [[maybe_unused]] const __sndr_t& __self, [[maybe_unused]] _Env&& __env) noexcept
  {
    if constexpr (_SetTag{} == set_value)
    {
      // When it completes successfully, the on(sch, sndr) sender completes where it
      // starts.
      return __call_or(execution::get_completion_domain<_SetTag>, default_domain{}, execution::get_scheduler(__env));
    }
    else
    {
      // When an on(sch, sndr) sender is connected with rcvr and started, three senders get
      // connected and started:
      //   1. The sender returned from schedule(sch).
      //   2. sndr (which sees that the current scheduler is sch).
      //   3. The sender returned from schedule(get_scheduler(get_env(rcvr))).
      //
      // Let Ds be a pack of the _SetTag completion domains of those senders that have
      // _SetTag completions and know their completion domain. The completion domain for _SetTag
      // of the on(sch, sndr) sender is common_type_t<Ds...>.
      constexpr bool __new_sch_has_compl = __has_completions_for<_SetTag, schedule_result_t<_Sch>, __fwd_env_t<_Env>>;
      constexpr bool __sndr_has_compl    = __has_completions_for<_SetTag, _Sndr, __env2_t<_Sch, _Env>>;
      constexpr bool __old_sch_has_compl =
        __has_completions_for<_SetTag, schedule_result_t<_OldSch>, __fwd_env_t<_Env>>;

      using __new_sch_domain_t = __call_result_or_t<get_completion_domain_t<_SetTag>, default_domain, _Sch>;
      using __old_sch_domain_t = __call_result_or_t<get_completion_domain_t<_SetTag>, default_domain, _OldSch>;
      using __sndr_domain_t =
        __type_call_or_q<__call_result_t, void, get_completion_domain_t<_SetTag>, env_of_t<_Sndr>, __env2_t<_Sch, _Env>>;

      using __domain_t =
        __type_call_or_q<::cuda::std::common_type_t,
                         void,
                         ::cuda::std::_If<__new_sch_has_compl, __new_sch_domain_t, __any_t>,
                         ::cuda::std::_If<__sndr_has_compl, __sndr_domain_t, __any_t>,
                         ::cuda::std::_If<__old_sch_has_compl, __old_sch_domain_t, __any_t>>;

      return ::cuda::std::_If<__same_as<__domain_t, __any_t>, void, __domain_t>();
    }
  }

  template <class _SetTag, class _Env>
  using __completion_scheduler_for_t =
    __unless_one_of_t<decltype(__sndr_t::__get_completion_scheduler_for(
                        _SetTag{}, ::cuda::std::declval<const __sndr_t&>(), ::cuda::std::declval<_Env>())),
                      void>;

  template <class _SetTag, class _Env>
  using __completion_domain_for_t =
    __unless_one_of_t<decltype(__sndr_t::__get_completion_domain_for(
                        _SetTag{}, ::cuda::std::declval<const __sndr_t&>(), ::cuda::std::declval<_Env>())),
                      void>;

  struct __attrs_t
  {
    template <class _SetTag, class _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<_SetTag>, _Env&& __env) const noexcept
      -> __completion_scheduler_for_t<_SetTag, _Env>
    {
      return __get_completion_scheduler_for(_SetTag{}, *this, static_cast<_Env&&>(__env));
    }

    template <class _SetTag, class _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<_SetTag>, _Env&&) const noexcept
      -> __completion_domain_for_t<get_completion_domain_t<_SetTag>, _Env>
    {
      return {};
    }

    template <class _SetTag, class _Env>
    _CCCL_API auto query(get_completion_scheduler_t<_SetTag>, _Env&&) const volatile = delete;

    template <class _SetTag, class _Env>
    _CCCL_API auto query(get_completion_domain_t<_SetTag>, _Env&&) const volatile = delete;

    // The completion behavior of `on(sch, sndr)` is the weaker of the
    // completion behaviors of the child sender, the new scheduler's sender,
    // and the old scheduler's sender.
    template <class _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t, _Env&&) const noexcept
    {
      using __old_sch_t = __call_result_or_t<get_scheduler_t, __not_a_scheduler, _Env>;
      return (execution::min) (execution::get_completion_behavior<schedule_result_t<_Sch>, _Env>(),
                               execution::get_completion_behavior<_Sndr, __env2_t<_Sch, _Env>>(),
                               execution::get_completion_behavior<schedule_result_t<__old_sch_t>, __fwd_env_t<_Env>>());
    }

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query, class... _Args)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query, _Args...>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query, _Args&&... __args) const
      noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query, _Args...>)
        -> __query_result_t<env_of_t<_Sndr>, _Query, _Args...>
    {
      return execution::get_env(__self_->__sndr_).query(_Query{}, static_cast<_Args&&>(__args)...);
    }

    const __sndr_t* __self_;
  };

public:
  using sender_concept = sender_t;

  _CCCL_NODEBUG_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return {__sndr_};
  }

  _CCCL_NO_UNIQUE_ADDRESS on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;
};

// This is the sender used for `on(sndr, sch, closure)` and `sndr | on(sch, closure)`.
template <class _Sch, class _Sndr, class _Closure>
struct _CCCL_TYPE_VISIBILITY_DEFAULT on_t::__sndr_t<_Sch, _Sndr, _Closure>
{
  using sender_concept = sender_t;

  _CCCL_NODEBUG_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Sndr>>
  {
    return __fwd_env(execution::get_env(__sndr_));
  }

  _CCCL_NO_UNIQUE_ADDRESS on_t __tag_;
  __closure_t<_Sch, _Closure> __sch_closure_;
  _Sndr __sndr_;
};

_CCCL_GLOBAL_CONSTANT on_t on{};

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<on_t::__sndr_t<_Sch, _Sndr>> = 3;

template <class _Sch, class _Sndr, class _Closure>
inline constexpr size_t structured_binding_size<on_t::__sndr_t<_Sch, _Sndr, _Closure>> = 3;

template <class _Sndr, class _NewSch, class _Env, class... _Closure>
inline constexpr size_t structured_binding_size<on_t::__lowered_sndr_t<_Sndr, _NewSch, _Env, _Closure...>> = 3;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_ON
