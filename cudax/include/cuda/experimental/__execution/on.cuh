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
#include <cuda/experimental/__execution/sndr_ref.cuh>
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
//! auto sndr = on(gpu_scheduler, some_computation);
//! auto [result] = sync_wait(std::move(sndr)).value();
//! @endcode
//!
//! ## Form 2: `on(sender, scheduler, closure)` or `sender | on(scheduler, closure)`
//!
//! Upon completion of the input sender, transfers execution to the specified scheduler's
//! execution resource, executes the closure with the sender's results, and then transfers
//! execution back to where the original sender completed.
//!
//! @code
//! auto sndr = some_computation | on(gpu_scheduler, then([](auto value) { /*...*/ }));
//! auto [result] = sync_wait(std::move(sndr)).value();
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
    operator()(_Sndr __sndr, _NewSch __new_sch, _OldSch __old_sch, _Closure&& __closure) const
    {
      return continues_on(static_cast<_Closure&&>(__closure)(continues_on(static_cast<_Sndr&&>(__sndr), __new_sch)),
                          __old_sch);
    }
  };

public:
  template <class _Sch, class _Sndr, class... _Closure>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sch, class _Closure>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t
  {
    template <class _Sndr>
    [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr __sndr) &&
    {
      return on_t{}(static_cast<_Sndr&&>(__sndr), __sch_, static_cast<_Closure&&>(__closure_));
    }

    template <class _Sndr>
    [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr __sndr) const&
    {
      return on_t{}(static_cast<_Sndr&&>(__sndr), __sch_, __closure_);
    }

    template <class _Sndr>
    [[nodiscard]] _CCCL_API friend constexpr auto operator|(_Sndr __sndr, __closure_t __self)
    {
      return on_t{}(static_cast<_Sndr&&>(__sndr), __self.__sch_, static_cast<_Closure&&>(__self.__closure_));
    }

    _Sch __sch_;
    _Closure __closure_;
  };

  _CCCL_TEMPLATE(class _Sch, class _Sndr)
  _CCCL_REQUIRES(__is_sender<_Sndr>)
  _CCCL_API constexpr auto operator()(_Sch __sch, _Sndr __sndr) const
  {
    return __sndr_t<_Sch, _Sndr>{{}, __sch, __sndr};
  }

  _CCCL_TEMPLATE(class _Sch, class _Closure)
  _CCCL_REQUIRES((!__is_sender<_Closure>) )
  _CCCL_API constexpr auto operator()(_Sch __sch, _Closure __closure) const
  {
    return __closure_t<_Sch, _Closure>{__sch, static_cast<_Closure&&>(__closure)};
  }

  template <class _Sndr, class _Sch, class _Closure>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr __sndr, _Sch __sch, _Closure __closure) const
  {
    using __sndr_t = on_t::__sndr_t<_Sch, _Sndr, _Closure>;
    return __sndr_t{{}, {__sch, static_cast<_Closure&&>(__closure)}, static_cast<_Sndr&&>(__sndr)};
  }

  template <class _Sndr, class _Env>
  [[nodiscard]] _CCCL_API static constexpr auto transform_sender(set_value_t, _Sndr&& __sndr, const _Env& __env)
  {
    auto&& [__ign, __data, __child] = __sndr;
    if constexpr (__is_scheduler<decltype(__data)>)
    {
      // The on(sch, sndr) case:
      auto __old_sch = __call_or(get_scheduler, __not_a_scheduler{}, __env);
      using __sndr_t = __lowered_sndr_t<decltype(__child), decltype(__data), decltype(__old_sch)>;
      static_assert(sender_for<__sndr_t, continues_on_t>);
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
  using sender_concept = sender_t;

  struct __attrs_t
  {
    template <class _Env>
    using __new_sndr_t = __call_result_t<__lower_sndr_fn, __sndr_ref<const _Sndr&>, _Sch, __scheduler_of_t<_Env>>;

    _CCCL_TEMPLATE(class _Tag, class _Env)
    _CCCL_REQUIRES(__queryable_with<env_of_t<__new_sndr_t<_Env>>, _Tag, _Env>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Tag, const _Env& __env) const
      noexcept(__nothrow_queryable_with<env_of_t<__new_sndr_t<_Env>>, _Tag, _Env>) -> decltype(auto)
    {
      auto __tmp_sndr = __lower_sndr_fn()(__sndr_ref(__self_->__sndr_), __self_->__sch_, get_scheduler(__env));
      return execution::get_env(__tmp_sndr).query(_Tag(), __env);
    }

    const __sndr_t* __self_;
  };

  _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return {__sndr_};
  }

  /*_CCCL_NO_UNIQUE_ADDRESS*/ on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;
};

// This is the sender used for `on(sndr, sch, closure)` and `sndr | on(sch, closure)`.
template <class _Sch, class _Sndr, class _Closure>
struct _CCCL_TYPE_VISIBILITY_DEFAULT on_t::__sndr_t<_Sch, _Sndr, _Closure>
{
  using sender_concept = sender_t;

  struct __attrs_t
  {
    template <class _Env>
    using __new_sndr_t =
      __call_result_t<__lower_sndr_fn, __sndr_ref<const _Sndr&>, _Sch, __scheduler_of_t<_Env>, const _Closure&>;

    _CCCL_TEMPLATE(class _Tag, class _Env)
    _CCCL_REQUIRES(__queryable_with<env_of_t<__new_sndr_t<_Env>>, _Tag, _Env>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Tag, const _Env& __env) const
      noexcept(__nothrow_queryable_with<env_of_t<__new_sndr_t<_Env>>, _Tag, _Env>) -> decltype(auto)
    {
      auto __tmp_sndr = __lower_sndr_fn()(
        __sndr_ref(__self_->__sndr_),
        __self_->__sch_closure_.__sch_,
        get_scheduler(__env),
        __self_->__sch_closure_.__closure_);
      return execution::get_env(__tmp_sndr).query(_Tag(), __env);
    }

    const __sndr_t* __self_;
  };

  _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return __attrs_t{this};
  }

  /*_CCCL_NO_UNIQUE_ADDRESS*/ on_t __tag_;
  __closure_t<_Sch, _Closure> __sch_closure_;
  _Sndr __sndr_;
};

_CCCL_GLOBAL_CONSTANT on_t on{};

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<on_t::__sndr_t<_Sch, _Sndr>> = 3;

template <class _Sch, class _Sndr, class _Closure>
inline constexpr size_t structured_binding_size<on_t::__sndr_t<_Sch, _Sndr, _Closure>> = 3;

template <class _Sndr, class _NewSch, class _OldSch, class... _Closure>
inline constexpr size_t structured_binding_size<on_t::__lowered_sndr_t<_Sndr, _NewSch, _OldSch, _Closure...>> = 3;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_ON
