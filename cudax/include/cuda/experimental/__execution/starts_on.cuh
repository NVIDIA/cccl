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
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/rcvr_with_env.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct starts_on_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Sch, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_base_t
  {
    // When the schedule sender completes, the receiver must start the child operation.
    // The type of the child operation depends on the type of the child sender. We don't
    // want sender types to be a part of a receiver's type because it can blow up the
    // length of the type name. So we indirect though a function pointer to start the
    // child operation.
    using __start_fn_t = void(__state_base_t*) noexcept;

    _Sch __sch_;
    _Rcvr __rcvr_;
    __start_fn_t* __start_fn_;
  };

  template <class _Sch, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr1_t
  {
    using receiver_concept = receiver_t;

    _CCCL_API void set_value() noexcept
    {
      // The scheduler operation completed successfully, and we are now executing on the
      // scheduler's execution context. Start the second operation.
      __state_->__start_fn_(__state_);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __err) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__err));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr_));
    }

    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_));
    }

    __state_base_t<_Sch, _Rcvr>* __state_;
  };

  // This is the environment type that is used by the receiver connected to the
  // child sender (__rcvr2_t below). It informs the child sender that it is being
  // started on the specified scheduler.
  template <class _Sch, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    [[nodiscard]] _CCCL_API constexpr auto query(get_scheduler_t) const noexcept -> _Sch
    {
      return __sch_;
    }

    [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_t) noexcept
    {
      return _CUDA_VSTD::__call_result_t<get_domain_t, _Sch>{};
    }

    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<_Env, _Query>)
    [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<_Env, _Query>)
      -> __query_result_t<_Env, _Query>
    {
      return __env_.query(_Query{});
    }

    _Sch __sch_;
    _Env __env_;
  };

  // This receiver is connected to the child sender.
  template <class _Sch, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr2_t
  {
    using receiver_concept = receiver_t;

    template <class... _Ts>
    _CCCL_API constexpr void set_value(_Ts&&... __ts) noexcept
    {
      execution::set_value(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Ts&&>(__ts)...);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __err) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__err));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr_));
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __env_t<_Sch, env_of_t<_Rcvr>>
    {
      return {__state_->__sch_, execution::get_env(__state_->__rcvr_)};
    }

    __state_base_t<_Sch, _Rcvr>* __state_;
  };

  template <class _Sch, class _CvSndr, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t : __state_base_t<_Sch, _Rcvr>
  {
    // FUTURE: put these in a variant
    using __opstate1_t = connect_result_t<schedule_result_t<_Sch&>, __rcvr1_t<_Sch, _Rcvr>>;
    using __opstate2_t = connect_result_t<_CvSndr, __rcvr2_t<_Sch, _Rcvr>>;

    _CCCL_API constexpr explicit __state_t(_Sch __sch, _CvSndr&& __sndr, _Rcvr __rcvr)
        : __state_base_t<_Sch, _Rcvr>{static_cast<_Sch&&>(__sch), static_cast<_Rcvr&&>(__rcvr), &__start_fn}
        , __opstate1_{execution::connect(execution::schedule(this->__sch_), __rcvr1_t<_Sch, _Rcvr>{this})}
        , __opstate2_{execution::connect(static_cast<_CvSndr&&>(__sndr), __rcvr2_t<_Sch, _Rcvr>{this})}
    {}

    _CCCL_API static void __start_fn(__state_base_t<_Sch, _Rcvr>* __state_base) noexcept
    {
      // This is the function that is called by the first operation state to start the second
      // operation state. It is a static function so that it can be used as a function pointer.
      auto* __state = static_cast<__state_t*>(__state_base);
      execution::start(__state->__opstate2_);
    }

    __opstate1_t __opstate1_;
    __opstate2_t __opstate2_;
  };

  template <class _Sch, class _CvSndr, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;

    _CCCL_API constexpr explicit __opstate_t(_Sch __sch, _CvSndr&& __sndr, _Rcvr __rcvr)
        : __state_{__sch, static_cast<_CvSndr&&>(__sndr), static_cast<_Rcvr&&>(__rcvr)}
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__state_.__opstate1_);
    }

    __state_t<_Sch, _CvSndr, _Rcvr> __state_;
  };

  template <class _Domain, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
  {
    [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(get_domain_override_t) noexcept
    {
      return _Domain{};
    }

    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query) const
      noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query>) -> __query_result_t<env_of_t<_Sndr>, _Query>
    {
      return execution::get_env(__sndr_).query(_Query{});
    }

    _Sndr const& __sndr_;
  };

public:
  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sch, class _Sndr>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sch __sch, _Sndr __sndr) const;
};

template <class _Sch, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT starts_on_t::__sndr_t
{
  using sender_concept                 = sender_t;
  using __domain_t _CCCL_NODEBUG_ALIAS = __query_result_or_t<_Sch, get_domain_t, default_domain>;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    using __sch_sndr _CCCL_NODEBUG_ALIAS   = schedule_result_t<_Sch>;
    using __child_sndr _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__copy_cvref_t<_Self, _Sndr>;
    _CUDAX_LET_COMPLETIONS(
      auto(__sndr_completions) = execution::get_completion_signatures<__child_sndr, __env_t<_Sch, _Env>...>())
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__sch_completions) = execution::get_completion_signatures<__sch_sndr, __fwd_env_t<_Env>...>())
      {
        return __sndr_completions + transform_completion_signatures(__sch_completions, __swallow_transform{});
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sch, _Sndr, _Rcvr>
  {
    return __opstate_t<_Sch, _Sndr, _Rcvr>{__sch_, static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Sch, const _Sndr&, _Rcvr>
  {
    return __opstate_t<_Sch, const _Sndr&, _Rcvr>{__sch_, __sndr_, static_cast<_Rcvr&&>(__rcvr)};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t<__domain_t, _Sndr>
  {
    return {__sndr_};
  }

  _CCCL_NO_UNIQUE_ADDRESS starts_on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;
};

template <class _Sch, class _Sndr>
[[nodiscard]] _CCCL_TRIVIAL_API constexpr auto starts_on_t::operator()(_Sch __sch, _Sndr __sndr) const
{
  using __sndr_t _CCCL_NODEBUG_ALIAS = starts_on_t::__sndr_t<_Sch, _Sndr>;
  return execution::transform_sender(
    execution::get_domain(__sch), __sndr_t{{}, static_cast<_Sch&&>(__sch), static_cast<_Sndr&&>(__sndr)});
}

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<starts_on_t::__sndr_t<_Sch, _Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT starts_on_t starts_on{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STARTS_ON
