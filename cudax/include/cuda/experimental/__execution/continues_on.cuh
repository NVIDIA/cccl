//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_CONTINUES_ON
#define __CUDAX_EXECUTION_CONTINUES_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/schedule_from.cuh>
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
  [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()() const noexcept
  {
    if constexpr (!__decay_copyable<_Ts...>)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, continues_on_t),
                                          _WHAT(_ARGUMENTS_ARE_NOT_DECAY_COPYABLE),
                                          _WITH_ARGUMENTS(_Ts...)>();
    }
    else if constexpr (!__nothrow_decay_copyable<_Ts...>)
    {
      return completion_signatures<_Tag(decay_t<_Ts>...), set_error_t(exception_ptr)>{};
    }
    else
    {
      return completion_signatures<_Tag(decay_t<_Ts>...)>{};
    }
  }
};
} // namespace __detail

struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t
{
  _CUDAX_SEMI_PRIVATE :
  struct __send_result_fn
  {
    template <class _Rcvr, class _Tag, class... _As>
    _CCCL_API constexpr void operator()(_Rcvr& __rcvr, _Tag, _As&... __args) const noexcept
    {
      // moves from lvalues here is intentional:
      _Tag{}(static_cast<_Rcvr&&>(__rcvr), static_cast<_As&&>(__args)...);
    }
  };

  struct __send_result_visitor
  {
    template <class _Rcvr, class _Tuple>
    _CCCL_API constexpr void operator()(_Rcvr& __rcvr, _Tuple& __tuple) const noexcept
    {
      ::cuda::std::__apply(__send_result_fn{}, __tuple, __rcvr);
    }
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
    using receiver_concept = receiver_t;

    _CCCL_API constexpr void set_value() noexcept
    {
      __state_->__result_.__visit(__send_result_visitor{}, __state_->__result_, __state_->__rcvr_);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __error) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__error));
    }

    _CCCL_API constexpr void set_stopped() noexcept
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
      using __tupl_t _CCCL_NODEBUG_ALIAS = ::cuda::std::__tuple<_Tag, decay_t<_As>...>;
      _CCCL_TRY
      {
        __state_->__result_.template __emplace<__tupl_t>(_Tag{}, static_cast<_As&&>(__as)...);
      }
      _CCCL_CATCH_ALL
      {
        // Avoid ODR-using this completion operation if this code path is not taken.
        if constexpr (!__nothrow_decay_copyable<_As...>)
        {
          execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), execution::current_exception());
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
      typename __completions_t::template __transform_q<::cuda::std::__decayed_tuple, __variant>;
    using __rcvr_t       = continues_on_t::__rcvr_t<_Rcvr, __results_t>;
    using __stash_rcvr_t = continues_on_t::__stash_rcvr_t<_Sch, _Rcvr, __results_t>;

    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Sch __sch, _Rcvr __rcvr)
        : __state_{{static_cast<_Rcvr&&>(__rcvr), {}}, execution::connect(schedule(__sch), __rcvr_t{&__state_})}
        , __opstate1_{execution::connect(static_cast<_CvSndr&&>(__sndr), __stash_rcvr_t{&__state_})}
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate1_);
    }

    __state_t<_Sch, _Rcvr, __results_t> __state_;
    connect_result_t<_CvSndr, __stash_rcvr_t> __opstate1_;
  };

public:
  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t;

  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sch>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t
  {
    template <class _Sndr>
    [[nodiscard]]
    _CCCL_API constexpr auto operator()(_Sndr __sndr) const -> __sndr_t<_Sch, __call_result_t<schedule_from_t, _Sndr>>
    {
      static_assert(__is_sender<_Sndr>);
      using __child_t = __call_result_t<schedule_from_t, _Sndr>;
      return __sndr_t<_Sch, __child_t>{{}, __sch_, schedule_from(static_cast<_Sndr&&>(__sndr))};
    }

    template <class _Sndr>
    [[nodiscard]]
    _CCCL_API constexpr friend auto operator|(_Sndr __sndr, __closure_t __clsur)
      -> __sndr_t<_Sch, __call_result_t<schedule_from_t, _Sndr>>
    {
      static_assert(__is_sender<_Sndr>);
      using __child_t = __call_result_t<schedule_from_t, _Sndr>;
      return __sndr_t<_Sch, __child_t>{{}, __clsur.__sch_, schedule_from(static_cast<_Sndr&&>(__sndr))};
    }

    _Sch __sch_;
  };

  template <class _Sch>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Sch __sch) const -> __closure_t<_Sch>
  {
    static_assert(__is_scheduler<_Sch>);
    return __closure_t<_Sch>{__sch};
  }

  template <class _Sch, class _Sndr>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr __sndr, _Sch __sch) const
    -> __sndr_t<_Sch, __call_result_t<schedule_from_t, _Sndr>>
  {
    static_assert(__is_sender<_Sndr>);
    static_assert(__is_scheduler<_Sch>);
    using __child_t = __call_result_t<schedule_from_t, _Sndr>;
    return __sndr_t<_Sch, __child_t>{{}, __sch, schedule_from(static_cast<_Sndr&&>(__sndr))};
  }
};

//! @brief The @c continues_on sender's attributes.
template <class _Sch, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t::__attrs_t
{
private:
  //! @brief Returns `true` when:
  //! - _SetTag is set_error_t, and
  //! - _Sndr has value completions, and
  //! - at least one of the value completions is not nothrow decay-copyable.
  //! In that case, error completions can come from the sender's value completions.
  template <class _SetTag, class... _Env>
  _CCCL_API static _CCCL_CONSTEVAL bool __has_decay_copy_errors() noexcept
  {
    if constexpr (__same_as<_SetTag, set_error_t>)
    {
      if constexpr (__has_completions_for<_Sndr, set_value_t, __fwd_env_t<_Env>...>)
      {
        using __completion_parts_t =
          __partitioned_completions_of_t<completion_signatures_of_t<_Sndr, __fwd_env_t<_Env>...>>;
        return !__completion_parts_t::__nothrow_decay_copyable::__values::value;
      }
    }
    return false;
  }

  const __sndr_t<_Sch, _Sndr>& __self_;

public:
  _CCCL_API constexpr explicit __attrs_t(const __sndr_t<_Sch, _Sndr>& __self) noexcept
      : __self_(__self)
  {}

  //! @brief Queries the completion scheduler for a given @c _SetTag.
  //! @tparam _SetTag The completion tag to query for.
  //! @tparam _Env The environment to consider when querying for the completion
  //! scheduler.
  //!
  //! @note If @c _SetTag is @c set_value_t, then we are in the happy path: everything
  //! succeeded and execution continues on @c _Sch.
  //!
  //! Otherwise, if @c _Sndr never completes with @c _SetTag, and either @c _SetTag is
  //! @c set_stopped_t or decay-copying @c _Sndr's value results cannot throw, then a
  //! @c _SetTag completion can only come from the scheduler's sender. In this case, return
  //! the scheduler's completion scheduler if it has one.
  //!
  //! Otherwise, if the scheduler's sender never completes with @c _SetTag, then a
  //! @c _SetTag completion can only come from the original sender, so return the
  //! original sender's completion scheduler.
  _CCCL_TEMPLATE(class _SetTag, class... _Env)
  _CCCL_REQUIRES((__same_as<_SetTag, set_value_t> || __never_completes_with<_Sndr, _SetTag, __fwd_env_t<_Env>...>)
                   _CCCL_AND(!__has_decay_copy_errors<_SetTag, _Env...>()))
  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<_SetTag>, const _Env&... __env) const noexcept
    -> __call_result_t<get_completion_scheduler_t<_SetTag>, _Sch, __fwd_env_t<_Env>...>
  {
    return get_completion_scheduler<_SetTag>(__self_.__sch_, __fwd_env(__env)...);
  }

  //! @overload
  _CCCL_TEMPLATE(class _SetTag, class... _Env)
  _CCCL_REQUIRES(__never_completes_with<schedule_result_t<_Sch>, _SetTag, __fwd_env_t<_Env>...>)
  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<_SetTag>, const _Env&... __env) const noexcept
    -> __call_result_t<get_completion_scheduler_t<_SetTag>, env_of_t<_Sndr>, __fwd_env_t<_Env>...>
  {
    return get_completion_scheduler<_SetTag>(get_env(__self_.__sndr_), __fwd_env(__env)...);
  }

  //! @brief Queries the completion domain for a given @c _SetTag.
  //! @tparam _SetTag The completion tag to query for.
  //! @tparam _Env The environment to consider when querying for the completion domain.
  //!
  //! @note If @c _SetTag is @c set_value_t, then we are in the happy path: everything
  //! succeeded and execution continues on @c _Sch.
  //!
  //! Otherwise, if @c _SetTag is @c set_stopped_t or if decay-copying @c _Sndr's value
  //! results cannot throw, then a @c _SetTag completion can happen on the sender's
  //! completion domain (if it has one) or the scheduler's completion domain (if it has
  //! one).
  //!
  //! @note Otherwise, @c _SetTag is @c set_error_t and decay-copying @c _Sndr's value
  //! results can throw, so error completions can also come from the sender's value
  //! completions.
  _CCCL_TEMPLATE(class _SetTag, class... _Env)
  _CCCL_REQUIRES(__same_as<_SetTag, set_value_t>)
  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<_SetTag>, const _Env&...) const noexcept
    -> __unless_one_of_t<__compl_domain_t<_SetTag, schedule_result_t<_Sch>, __fwd_env_t<_Env>...>, __not_a_domain>
  {
    return {};
  }

  //! @overload
  _CCCL_TEMPLATE(class _SetTag, class... _Env)
  _CCCL_REQUIRES((!__same_as<_SetTag, set_value_t>) _CCCL_AND(!__has_decay_copy_errors<_SetTag, _Env...>()))
  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<_SetTag>, const _Env&...) const noexcept
    -> __unless_one_of_t<__common_domain_t<__compl_domain_t<_SetTag, _Sndr, __fwd_env_t<_Env>...>,
                                           __compl_domain_t<_SetTag, schedule_result_t<_Sch>, __fwd_env_t<_Env>...>>,
                         __not_a_domain>
  {
    return {};
  }

  //! @overload
  _CCCL_TEMPLATE(class _SetTag, class... _Env)
  _CCCL_REQUIRES((__has_decay_copy_errors<_SetTag, _Env...>()))
  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<_SetTag>, const _Env&...) const noexcept
    -> __unless_one_of_t<__common_domain_t<__compl_domain_t<_SetTag, _Sndr, __fwd_env_t<_Env>...>,
                                           __compl_domain_t<_SetTag, schedule_result_t<_Sch>, __fwd_env_t<_Env>...>,
                                           __compl_domain_t<set_value_t, _Sndr, __fwd_env_t<_Env>...>>,
                         __not_a_domain>
  {
    return {};
  }

  //! @brief Queries the completion behavior of the combined sender.
  //! @tparam _Env The environment to consider when querying for the completion behavior.
  //! @note The completion behavior is the minimum between the scheduler's sender and
  //! the original sender.
  template <class... _Env>
  [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t, const _Env&...) const noexcept
  {
    return (execution::min) (execution::get_completion_behavior<schedule_result_t<_Sch>, __fwd_env_t<_Env>...>(),
                             execution::get_completion_behavior<_Sndr, _Env...>());
  }

  //! @brief Forwards other queries to the underlying sender's environment.
  //! @pre @c _Tag is a forwarding query but not a completion query.
  _CCCL_TEMPLATE(class _Tag, class... _Args)
  _CCCL_REQUIRES(__forwarding_query<_Tag> _CCCL_AND(!__is_completion_query<_Tag>)
                   _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Tag, _Args...>)
  [[nodiscard]] _CCCL_API constexpr auto query(_Tag, _Args&&... __args) const
    noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Tag, _Args...>)
      -> __query_result_t<env_of_t<_Sndr>, _Tag, _Args...>
  {
    return get_env(__self_.__sndr_).query(_Tag{}, static_cast<_Args&&>(__args)...);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// continues_on sender
template <class _Sch, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t::__sndr_t
{
  using sender_concept = sender_t;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__child_completions) = get_child_completion_signatures<_Self, _Sndr, _Env...>())
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__sch_completions) = execution::get_completion_signatures<schedule_result_t<_Sch>, __fwd_env_t<_Env>...>())
      {
        // The scheduler contributes error and stopped completions.
        return concat_completion_signatures(
          transform_completion_signatures(__sch_completions, __swallow_transform{}),
          transform_completion_signatures(
            __child_completions, __detail::__decay_args<set_value_t>{}, __detail::__decay_args<set_error_t>{}));
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sch, _Sndr, _Rcvr>
  {
    return __opstate_t<_Sch, _Sndr, _Rcvr>{static_cast<_Sndr&&>(__sndr_), __sch_, static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Sch, const _Sndr&, _Rcvr>
  {
    return __opstate_t<_Sch, const _Sndr&, _Rcvr>{__sndr_, __sch_, static_cast<_Rcvr&&>(__rcvr)};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t<_Sch, _Sndr>
  {
    return __attrs_t<_Sch, _Sndr>(*this);
  }

  /*_CCCL_NO_UNIQUE_ADDRESS*/ continues_on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;
};

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<continues_on_t::__sndr_t<_Sch, _Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT continues_on_t continues_on{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_CONTINUES_ON
