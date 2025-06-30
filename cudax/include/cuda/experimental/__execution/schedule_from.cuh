//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_SCHEDULE_FROM
#define __CUDAX_EXECUTION_SCHEDULE_FROM

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __detail
{
template <class _Tag>
struct __decay_args
{
  template <class... _Ts>
  [[nodiscard]] _CCCL_TRIVIAL_API _CCCL_CONSTEVAL auto operator()() const noexcept
  {
    if constexpr (!__decay_copyable<_Ts...>)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, schedule_from_t),
                                          _WHAT(_ARGUMENTS_ARE_NOT_DECAY_COPYABLE),
                                          _WITH_ARGUMENTS(_Ts...)>();
    }
    else if constexpr (!__nothrow_decay_copyable<_Ts...>)
    {
      return completion_signatures<_Tag(_CUDA_VSTD::decay_t<_Ts>...), set_error_t(::std::exception_ptr)>{};
    }
    else
    {
      return completion_signatures<_Tag(_CUDA_VSTD::decay_t<_Ts>...)>{};
    }
  }
};

// A base class for both schedule_from_t::__sndr_t and continues_on_t::__sndr_t
template <class _Tag, class _Sch, class _Sndr>
struct __transfer_sndr_t
{
  using sender_concept = sender_t;

  // For schedule_from, the domain to use when transforming the sender is the same as the
  // scheduler's domain, or default_domain if it does not define one. For continues_on,
  // the transformation domain is the same as the predecessor sender's domain, if it
  // defines one.
  using __sched_domain_t _CCCL_NODEBUG_ALIAS = __query_result_or_t<_Sch, get_domain_t, default_domain>;
  using __sndr_domain_t _CCCL_NODEBUG_ALIAS  = __early_domain_of_t<_Sndr, __nil>;
  using __late_domain_t _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::_If<_CUDA_VSTD::is_same_v<_Tag, schedule_from_t>, __sched_domain_t, __sndr_domain_t>;

  // see SCHED-ATTRS here: https://eel.is/c++draft/exec#snd.expos-6
  struct __attrs_t
  {
    template <class _SetTag>
    _CCCL_API constexpr auto query(get_completion_scheduler_t<_SetTag>) const = delete;

    _CCCL_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> _Sch
    {
      return __self_->__sch_;
    }

    // Both schedule_from and continues_on senders complete on the scheduler's domain.
    [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_t) noexcept -> __sched_domain_t
    {
      return {};
    }

    // schedule_from and continues_on have special rules for the domain used to transform
    // the sender.
    _CCCL_TEMPLATE(class _LateDomain = __late_domain_t)
    _CCCL_REQUIRES((!_CUDA_VSTD::same_as<_LateDomain, __nil>) )
    [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_override_t) noexcept -> _LateDomain
    {
      return {};
    }

    // The following overload will not be considered when _Query is get_domain_override_t
    // because get_domain_override_t is not a forwarding query.
    _CCCL_TEMPLATE(class _Query)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query) const
      noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query>) -> __query_result_t<env_of_t<_Sndr>, _Query>
    {
      return execution::get_env(__self_->__sndr_).query(_Query{});
    }

    const __transfer_sndr_t* __self_;
  };

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__child_completions) = get_child_completion_signatures<_Self, _Sndr, _Env...>())
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__sch_completions) = execution::get_completion_signatures<schedule_result_t<_Sch>, _Env...>())
      {
        // The scheduler contributes error and stopped completions.
        return concat_completion_signatures(
          transform_completion_signatures(__sch_completions, __swallow_transform{}),
          transform_completion_signatures(
            __child_completions, __decay_args<set_value_t>{}, __decay_args<set_error_t>{}));
      }
    }

    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return __attrs_t{this};
  }

  _CCCL_NO_UNIQUE_ADDRESS _Tag __tag_;
  _Sch __sch_;
  _Sndr __sndr_;
};

} // namespace __detail

struct _CCCL_TYPE_VISIBILITY_DEFAULT schedule_from_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class... _As>
  using __set_value_tuple_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__tuple<set_value_t, _CUDA_VSTD::decay_t<_As>...>;

  template <class _Error>
  using __set_error_tuple_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__tuple<set_error_t, _CUDA_VSTD::decay_t<_Error>>;

  using __set_stopped_tuple_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__tuple<set_stopped_t>;

  struct __send_result_fn
  {
    template <class _Rcvr, class _Tag, class... _As>
    _CCCL_API constexpr void operator()(_Rcvr& __rcvr, _Tag, _As&&... __args) const noexcept
    {
      _Tag{}(static_cast<_Rcvr&&>(__rcvr), static_cast<_As&&>(__args)...);
    }
  };

  template <class _Rcvr>
  struct __send_result_visitor
  {
    template <class _Tuple>
    _CCCL_API constexpr void operator()(_Tuple&& __tuple) const noexcept
    {
      _CUDA_VSTD::__apply(__send_result_fn{}, static_cast<_Tuple&&>(__tuple), __rcvr_);
    }

    _Rcvr& __rcvr_;
  };

  template <class _Rcvr, class _Results>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_base_t
  {
    _Rcvr __rcvr_;
    _Results __result_;
  };

  // This receiver is connected to the scheduler. It forwards the results of the child sender,
  // which are stored in a variant, to the parent receiver.
  template <class _Rcvr, class _Results>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept _CCCL_NODEBUG_ALIAS = receiver_t;

    template <class _Tag, class... _As>
    _CCCL_API constexpr void operator()(_Tag, _As&... __as) noexcept
    {
      _Tag{}(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_As&&>(__as)...);
    }

    _CCCL_API constexpr void set_value() noexcept
    {
      __state_->__result_.__visit(__send_result_visitor<_Rcvr>{__state_->__rcvr_}, __state_->__result_);
    }

    template <class _Error>
    _CCCL_TRIVIAL_API constexpr void set_error(_Error&& __error) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__error));
    }

    _CCCL_TRIVIAL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr_));
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_));
    }

    __state_base_t<_Rcvr, _Results>* __state_;
  };

  template <class _Sch, class _Rcvr, class _Results>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t : __state_base_t<_Rcvr, _Results>
  {
    connect_result_t<schedule_result_t<_Sch>, __rcvr_t<_Rcvr, _Results>> __opstate2_;
  };

  // This receiver is connected to the child sender. It stashes the sender's results into
  // a variant.
  template <class _Sch, class _Rcvr, class _Results>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __stash_rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class _Tag, class... _As>
    _CCCL_API void __set_result(_Tag, _As&&... __as) noexcept
    {
      using __tupl_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__tuple<_Tag, _CUDA_VSTD::decay_t<_As>...>;
      if constexpr (__nothrow_decay_copyable<_As...>)
      {
        __state_->__result_.template __emplace<__tupl_t>(_Tag{}, static_cast<_As&&>(__as)...);
      }
      else
      {
        _CCCL_TRY
        {
          __state_->__result_.template __emplace<__tupl_t>(_Tag{}, static_cast<_As&&>(__as)...);
        }
        _CCCL_CATCH_ALL
        {
          execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), ::std::current_exception());
        }
      }
    }

    template <class... _As>
    _CCCL_API void set_value(_As&&... __as) noexcept
    {
      __set_result(set_value_t{}, static_cast<_As&&>(__as)...);
      execution::start(__state_->__opstate2_);
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __error) noexcept
    {
      __set_result(set_error_t{}, static_cast<_Error&&>(__error));
      execution::start(__state_->__opstate2_);
    }

    _CCCL_API void set_stopped() noexcept
    {
      __set_result(set_stopped_t{});
      execution::start(__state_->__opstate2_);
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_));
    }

    __state_t<_Sch, _Rcvr, _Results>* __state_;
  };

  template <class _Sch, class _CvSndr, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept             = operation_state_t;
    using __completions_t _CCCL_NODEBUG_ALIAS = completion_signatures_of_t<_CvSndr, __fwd_env_t<env_of_t<_Rcvr>>>;
    using __results_t _CCCL_NODEBUG_ALIAS =
      typename __completions_t::template __transform_q<_CUDA_VSTD::__decayed_tuple, __variant>;
    using __rcvr_t       = schedule_from_t::__rcvr_t<_Rcvr, __results_t>;
    using __stash_rcvr_t = schedule_from_t::__stash_rcvr_t<_Sch, _Rcvr, __results_t>;

    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Sch __sch, _Rcvr __rcvr)
        : __state_{{static_cast<_Rcvr&&>(__rcvr), {}}, execution::connect(schedule(__sch), __rcvr_t{&__state_})}
        , __opstate1_{execution::connect(static_cast<_CvSndr&&>(__sndr), __stash_rcvr_t{&__state_})}
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate1_);
    }

    __state_t<_Sch, _Rcvr, __results_t> __state_;
    connect_result_t<_CvSndr, __stash_rcvr_t> __opstate1_;
  };

public:
  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t : __detail::__transfer_sndr_t<schedule_from_t, _Sch, _Sndr>
  {
    template <class _Rcvr>
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sch, _Sndr, _Rcvr>
    {
      return __opstate_t<_Sch, _Sndr, _Rcvr>{
        static_cast<_Sndr&&>(this->__sndr_), this->__sch_, static_cast<_Rcvr&&>(__rcvr)};
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Sch, const _Sndr&, _Rcvr>
    {
      return __opstate_t<_Sch, const _Sndr&, _Rcvr>{this->__sndr_, this->__sch_, static_cast<_Rcvr&&>(__rcvr)};
    }
  };

  template <class _Sch, class _Sndr>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(_Sch __sch, _Sndr __sndr) const
  {
    static_assert(__is_sender<_Sndr>);
    static_assert(__is_scheduler<_Sch>);
    // schedule_from always dispatches based on the domain of the scheduler
    return transform_sender(get_domain(__sch), __sndr_t<_Sch, _Sndr>{{{}, __sch, static_cast<_Sndr&&>(__sndr)}});
  }
};

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<schedule_from_t::__sndr_t<_Sch, _Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT schedule_from_t schedule_from{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_SCHEDULE_FROM
