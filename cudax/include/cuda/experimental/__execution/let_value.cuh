//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_LET_VALUE
#define __CUDAX_EXECUTION_LET_VALUE

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
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/auto_cast.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/rcvr_with_env.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Query>
_CCCL_CONCEPT __forwarding_let_query =
  __forwarding_query<_Query> && //
  __none_of<_Query,
            get_completion_domain_t<set_value_t>,
            get_completion_domain_t<set_error_t>,
            get_completion_domain_t<set_stopped_t>>;

struct _CCCL_TYPE_VISIBILITY_DEFAULT __let_t
{
  template <class _LetTag>
  static ::cuda::std::__undefined<_LetTag> __set_tag;

  template <class _LetTag>
  using __set_tag_for_t = decltype(_LIBCUDACXX_AUTO_CAST(__set_tag<_LetTag>));

  //! @brief Computes the type of a variant of tuples to hold the results of the
  //! predecessor sender.
  template <class _SetTag, class _Completions, class _Env>
  using __sndr1_results_t _CCCL_NODEBUG_ALIAS =
    __gather_completion_signatures<_Completions, _SetTag, ::cuda::std::__decayed_tuple, __variant>;

  // This environment is part of the receiver used to connect the secondary sender.
  template <class _SetTag, class _Attrs, class... _Env>
  _CCCL_API static constexpr auto __mk_env2(const _Attrs& __attrs, const _Env&... __env) noexcept
  {
    if constexpr (__callable<get_completion_scheduler_t<_SetTag>, const _Attrs&, const _Env&...>)
    {
      return __mk_sch_env(get_completion_scheduler<_SetTag>(__attrs, __env...), __env...);
    }
    else if constexpr (__callable<get_completion_domain_t<_SetTag>, const _Attrs&, const _Env&...>)
    {
      return prop{get_domain, get_completion_domain<_SetTag>(__attrs, __env...)};
    }
    else
    {
      return env{};
    }
  }

  template <class _SetTag, class _Attrs, class... _Env>
  using __env2_t _CCCL_NODEBUG_ALIAS =
    decltype(__let_t::__mk_env2<_SetTag>(::cuda::std::declval<_Attrs>(), ::cuda::std::declval<_Env>()...));

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr2_fn
  {
    template <class... _As>
    using __call _CCCL_NODEBUG_ALIAS = __call_result_t<_Fn, decay_t<_As>&...>;
  };

  template <class _Rcvr, class _Env2>
  struct __sndr2_rcvr_t : __rcvr_ref_t<__rcvr_with_env_t<_Rcvr, _Env2>>
  {
    using __base_t = __rcvr_ref_t<__rcvr_with_env_t<_Rcvr, _Env2>>;

    _CCCL_NODEBUG_API explicit constexpr __sndr2_rcvr_t(__rcvr_with_env_t<_Rcvr, _Env2>& __rcvr) noexcept
        : __base_t(__ref_rcvr(__rcvr))
    {}
  };

  template <class _Fn, class _Rcvr, class _Env2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_base_t
  {
    //! @brief For a given set of result datums, compute the type of the secondary
    //! sender's operation state.
    template <class... _As>
    using __sndr2_opstate_fn _CCCL_NODEBUG_ALIAS =
      connect_result_t<::cuda::std::__type_call<__sndr2_fn<_Fn>, _As...>, __sndr2_rcvr_t<_Rcvr, _Env2>>;

    __rcvr_with_env_t<_Rcvr, _Env2> __rcvr_;
    _Fn __fn_;
  };

  template <class _SetTag, class _Fn, class _Rcvr, class _Env2, class _Completions>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t : __state_base_t<_Fn, _Rcvr, _Env2>
  {
    using __sndr2_opstate_t _CCCL_NODEBUG_ALIAS =
      __gather_completion_signatures<_Completions,
                                     _SetTag,
                                     __state_t::__state_base_t::template __sndr2_opstate_fn,
                                     __variant>;

    __sndr1_results_t<_SetTag, _Completions, __fwd_env_t<env_of_t<_Rcvr>>> __result_{};
    __sndr2_opstate_t __opstate2_{};
  };

  //! @brief This is the receiver that gets connected to the predecessor sender. It caches
  //! the results of the predecessor and then calls the user-provided function with them
  //! to produce the secondary sender, which it then connects and starts.
  template <class _SetTag, class _Fn, class _Rcvr, class _Env2, class _Completions>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr1_rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _As>
    _CCCL_API void __complete(_SetTag, _As&&... __as) noexcept
    {
      _CCCL_TRY
      {
        // Store the results so the lvalue refs we pass to the function will be valid for
        // the duration of the async op.
        auto& __tupl =
          __state_->__result_.template __emplace<::cuda::std::__decayed_tuple<_As...>>(static_cast<_As&&>(__as)...);

        // Call the function with the results and connect the resulting sender, storing
        // the operation state in __state_->__opstate2_.
        auto& __next_op = __state_->__opstate2_.__emplace_from(
          execution::connect,
          ::cuda::std::__apply(static_cast<_Fn&&>(__state_->__fn_), __tupl),
          __sndr2_rcvr_t(__state_->__rcvr_));
        execution::start(__next_op);
      }
      _CCCL_CATCH_ALL
      {
        execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_.__base()), ::std::current_exception());
      }
    }

    template <class _Tag, class... _As>
    _CCCL_API void __complete(_Tag, _As&&... __as) noexcept
    {
      // Forward the completion to the receiver unchanged.
      _Tag{}(static_cast<_Rcvr&&>(__state_->__rcvr_.__base()), static_cast<_As&&>(__as)...);
    }

    template <class... _As>
    _CCCL_API void set_value(_As&&... __as) noexcept
    {
      __complete(execution::set_value, static_cast<_As&&>(__as)...);
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __error) noexcept
    {
      __complete(execution::set_error, static_cast<_Error&&>(__error));
    }

    _CCCL_API void set_stopped() noexcept
    {
      __complete(execution::set_stopped);
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_.__base()));
    }

    __state_t<_SetTag, _Fn, _Rcvr, _Env2, _Completions>* __state_;
  };

  //! @brief The `let_(value|error|stopped)` operation state.
  //! @tparam _CvSndr The cvref-qualified predecessor sender type.
  //! @tparam _Fn The user-provided function to be called with the result datums of the
  //! predecessor sender.
  //! @tparam _Rcvr The receiver connected to the `let_(value|error|stopped)`
  //! sender.
  template <class _SetTag, class _CvSndr, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __completions_t         = completion_signatures_of_t<_CvSndr, __fwd_env_t<env_of_t<_Rcvr>>>;
    using __env2_t                = __let_t::__env2_t<_SetTag, env_of_t<_CvSndr>, env_of_t<_Rcvr>>;
    using __sndr1_rcvr_t          = __sndr1_rcvr_t<_SetTag, _Fn, _Rcvr, __env2_t, __completions_t>;

    _CCCL_API constexpr explicit __opstate_t(_CvSndr& __sndr, _Fn& __fn, _Rcvr& __rcvr, __env2_t&& __env2) noexcept(
      __nothrow_decay_copyable<_Fn, _Rcvr, __env2_t> && __nothrow_connectable<_CvSndr, __sndr1_rcvr_t>)
        : __state_{{{static_cast<_Rcvr&&>(__rcvr), static_cast<__env2_t&&>(__env2)}, static_cast<_Fn&&>(__fn)}}
        , __opstate1_(execution::connect(static_cast<_CvSndr&&>(__sndr), __sndr1_rcvr_t{&__state_}))
    {}

    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Fn __fn, _Rcvr __rcvr) noexcept(
      __nothrow_decay_copyable<_Fn, _Rcvr, __env2_t> && __nothrow_connectable<_CvSndr, __sndr1_rcvr_t>)
        : __opstate_t(
            __sndr, __fn, __rcvr, __let_t::__mk_env2<_SetTag>(execution::get_env(__sndr), execution::get_env(__rcvr)))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate1_);
    }

    __state_t<_SetTag, _Fn, _Rcvr, __env2_t, __completions_t> __state_;
    connect_result_t<_CvSndr, __sndr1_rcvr_t> __opstate1_;
  };

  template <class _SetTag, class _Fn, class _Env2, class... _Env>
  struct __domain_transform_fn
  {
    template <class... _As>
    using __call _CCCL_NODEBUG_ALIAS =
      __call_result_or_t<get_completion_domain_t<set_value_t>,
                         default_domain,
                         env_of_t<::cuda::std::__type_call<__sndr2_fn<_Fn>, _As...>>,
                         __join_env_t<_Env2, const _Env&...>>;
  };

  struct __domain_reduce_fn
  {
    template <class... _Domains>
    using __call _CCCL_NODEBUG_ALIAS = ::cuda::std::
      _If<sizeof...(_Domains) == 0, default_domain, __type_call_or_q<::cuda::std::common_type_t, void, _Domains...>>;
  };

  template <class _SetTag, class _Sndr, class _Fn, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto __get_completion_domain() noexcept
  {
    // we can know the completion domain for non-dependent senders
    using __completions_t = completion_signatures_of_t<_Sndr, _Env...>;
    if constexpr (__valid_completion_signatures<__completions_t>)
    {
      using __env2_t = __let_t::__env2_t<_SetTag, env_of_t<_Sndr>, _Env...>;
      return __gather_completion_signatures<__completions_t,
                                            _SetTag,
                                            __domain_transform_fn<_SetTag, _Fn, __env2_t, _Env...>::template __call,
                                            __domain_reduce_fn::template __call>();
    }
  }

  template <class _SetTag, class _Sndr, class _Fn, class... _Env>
  using __completion_domain_of_t _CCCL_NODEBUG_ALIAS =
    __unless_one_of_t<decltype(__let_t::__get_completion_domain<_SetTag, _Sndr, _Fn, _Env...>()), void>;

  template <class _LetTag, class _Fn, class _Env2, class... _Env>
  struct __transform_args_fn
  {
    template <class... _Ts>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()() const
    {
      if constexpr (!__decay_copyable<_Ts...>)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _LetTag),
                                            _WHAT(_ARGUMENTS_ARE_NOT_DECAY_COPYABLE),
                                            _WITH_ARGUMENTS(_Ts...)>();
      }
      else if constexpr (!::cuda::std::__type_callable<__sndr2_fn<_Fn>, _Ts...>::value)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _LetTag),
                                            _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                            _WITH_FUNCTION(_Fn),
                                            _WITH_ARGUMENTS(decay_t<_Ts> & ...)>();
      }
      else
      {
        using __sndr2_t = ::cuda::std::__type_call<__sndr2_fn<_Fn>, _Ts...>;
        if constexpr (!sender<__sndr2_t>)
        {
          return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _LetTag),
                                              _WHAT(_FUNCTION_MUST_RETURN_A_SENDER),
                                              _WITH_FUNCTION(_Fn),
                                              _WITH_ARGUMENTS(decay_t<_Ts> & ...),
                                              _WITH_RETURN_TYPE(__sndr2_t)>();
        }
        else
        {
          // The function is callable with the arguments and returns a sender, but we do
          // not know whether connect will throw.
          return execution::get_completion_signatures<__sndr2_t, __join_env_t<_Env2, _Env...>>() + __eptr_completion();
        }
      }
    }
  };

  template <class _Fn, class... _Env>
  struct __completion_behavior_transform_fn
  {
    template <class _Tag, class... _Ts>
    [[nodiscard]] _CCCL_API constexpr auto operator()(_Tag (*)(_Ts...)) const noexcept
    {
      if constexpr (::cuda::std::__type_callable<__sndr2_fn<_Fn>, _Ts...>::value)
      {
        using __sndr2_t = ::cuda::std::__type_call<__sndr2_fn<_Fn>, _Ts...>;
        return execution::get_completion_behavior<__sndr2_t, _Env...>();
      }
      else
      {
        return completion_behavior::unknown;
      }
    }
  };

  template <class _LetTag, class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _LetTag, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_HIDDEN __closure_t // hidden visibility because member __fn_ is hidden if it is an
                                                  // extended (host/device) lambda
  {
    template <class _Sndr>
    [[nodiscard]] _CCCL_NODEBUG_API auto operator()(_Sndr __sndr) const -> __call_result_t<_LetTag, _Sndr, _Fn>
    {
      return _LetTag{}(static_cast<_Sndr&&>(__sndr), __fn_);
    }

    template <class _Sndr>
    [[nodiscard]] _CCCL_NODEBUG_API friend auto operator|(_Sndr __sndr, const __closure_t& __self)
      -> __call_result_t<_LetTag, _Sndr, _Fn>
    {
      return _LetTag{}(static_cast<_Sndr&&>(__sndr), __self.__fn_);
    }

    _Fn __fn_;
  };
};

template <class _LetTag>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __let_base_t : __let_t
{
  //! @brief The `let_(value|error|stopped)` sender.
  //! @tparam _Sndr The predecessor sender.
  //! @tparam _Fn The function to be called when the predecessor sender
  //! completes.
  template <class _Sndr, class _Fn>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr, _Fn __fn) const;

  template <class _Fn>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Fn __fn) const noexcept;
};

struct let_value_t : __let_base_t<let_value_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

struct let_error_t : __let_base_t<let_error_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

struct let_stopped_t : __let_base_t<let_stopped_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

template <class _LetTag, class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __let_t::__sndr_t
{
  using sender_concept = sender_t;
  using __set_tag_t    = __set_tag_for_t<_LetTag>;

  // the env of the receiver used to connect the secondary sender
  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__child_completions) = get_child_completion_signatures<_Self, _Sndr, _Env...>())
    {
      // This is part of the environment of the receiver used to connect the secondary sender:
      using __env2_t                = __let_t::__env2_t<__set_tag_t, env_of_t<_Sndr>, _Env...>;
      constexpr auto __transform_fn = __transform_args_fn<_LetTag, _Fn, __env2_t, _Env...>{};

      if constexpr (!__is_instantiable_with<__completion_domain_of_t, __set_tag_t, _Sndr, _Fn, _Env...>
                    && (sizeof...(_Env) != 0))
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _LetTag),
                                            _WHAT(_FUNCTION_MUST_RETURN_SENDERS_THAT_ALL_COMPLETE_IN_A_COMMON_DOMAIN),
                                            _WITH_FUNCTION(_Fn)>();
      }
      else if constexpr (__set_tag_t{} == execution::set_value)
      {
        return transform_completion_signatures(__child_completions, __transform_fn);
      }
      else if constexpr (__set_tag_t{} == execution::set_error)
      {
        return transform_completion_signatures(__child_completions, {}, __transform_fn);
      }
      else
      {
        return transform_completion_signatures(__child_completions, {}, {}, __transform_fn);
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) && noexcept(
    __nothrow_constructible<__opstate_t<__set_tag_t, _Sndr, _Fn, _Rcvr>, _Sndr, _Fn, _Rcvr>)
    -> __opstate_t<__set_tag_t, _Sndr, _Fn, _Rcvr>
  {
    return __opstate_t<__set_tag_t, _Sndr, _Fn, _Rcvr>(
      static_cast<_Sndr&&>(__sndr_), static_cast<_Fn&&>(__fn_), static_cast<_Rcvr&&>(__rcvr));
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& noexcept(
    __nothrow_constructible<__opstate_t<__set_tag_t, const _Sndr&, _Fn, _Rcvr>, const _Sndr&, const _Fn&, _Rcvr>)
    -> __opstate_t<__set_tag_t, const _Sndr&, _Fn, _Rcvr>
  {
    return __opstate_t<__set_tag_t, const _Sndr&, _Fn, _Rcvr>(__sndr_, __fn_, static_cast<_Rcvr&&>(__rcvr));
  }

  // BUGBUG: think harder about the let_value sender attributes. forwarding queries to the
  // child sender seems questionable at best.
  struct __attrs_t
  {
    template <class _Tag>
    _CCCL_API constexpr auto query(get_completion_scheduler_t<_Tag>) const = delete;

    template <class... _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<__set_tag_t>, const _Env&...) const noexcept
      -> __unless_one_of_t<__completion_domain_of_t<__set_tag_t, _Sndr, _Fn, _Env...>, __nil>
    {
      return {};
    }

    template <class... _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t, const _Env&...) const noexcept
    {
      if constexpr (sender_in<_Sndr, _Env...>)
      {
        // The completion behavior of let_value(sndr, fn) is the weakest completion
        // behavior of sndr and all the senders that fn can potentially produce. (MSVC
        // needs the constexpr computation broken up, hence the local variables.)
        constexpr auto __completions =
          execution::get_completion_signatures<_Sndr, __fwd_env_t<_Env>...>().select(__set_tag_t{});
        constexpr auto __behavior =
          __completions.transform_reduce(__completion_behavior_transform_fn<_Fn, _Env...>{}, execution::min);
        return (execution::min) (execution::get_completion_behavior<_Sndr, __fwd_env_t<_Env>...>(), __behavior);
      }
      else
      {
        return completion_behavior::unknown;
      }
    }

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query, class... _Args)
    _CCCL_REQUIRES(__forwarding_let_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query, _Args...>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query, _Args&&... __args) const
      noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query, _Args...>)
        -> __query_result_t<env_of_t<_Sndr>, _Query, _Args...>
    {
      return execution::get_env(__sndr_).query(_Query{}, static_cast<_Args&&>(__args)...);
    }

    const _Sndr& __sndr_;
  };

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return {__sndr_};
  }

  _CCCL_NO_UNIQUE_ADDRESS _LetTag __tag_;
  _Fn __fn_;
  _Sndr __sndr_;
};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_value_t::__sndr_t : __let_t::__sndr_t<let_value_t, _Sndr, _Fn>
{};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_error_t::__sndr_t : __let_t::__sndr_t<let_error_t, _Sndr, _Fn>
{};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_stopped_t::__sndr_t : __let_t::__sndr_t<let_stopped_t, _Sndr, _Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_value_t::__closure_t : __let_t::__closure_t<let_value_t, _Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_error_t::__closure_t : __let_t::__closure_t<let_error_t, _Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_stopped_t::__closure_t : __let_t::__closure_t<let_stopped_t, _Fn>
{};

template <class... _Sndr>
using __all_non_dependent_t = ::cuda::std::__fold_and<(!dependent_sender<_Sndr>) ...>;

template <class _LetTag>
template <class _Sndr, class _Fn>
[[nodiscard]] _CCCL_NODEBUG_API constexpr auto __let_base_t<_LetTag>::operator()(_Sndr __sndr, _Fn __fn) const
{
  using __sndr_t   = typename _LetTag::template __sndr_t<_Sndr, _Fn>;
  using __domain_t = __early_domain_of_t<_Sndr>;
  // If the incoming sender is non-dependent, we can check the completion signatures of
  // the composed sender immediately.
  if constexpr (!dependent_sender<_Sndr>)
  {
    // Although the input sender is not dependent, the sender(s) returned from the
    // function might be. Only do eager type-checking if all the possible senders returned
    // by the function are non-dependent. If any of them is dependent, we will defer the
    // type-checking to the point where the sender is connected.
    using __completions_t = completion_signatures_of_t<_Sndr>;
    constexpr bool __all_non_dependent =
      __gather_completion_signatures<__completions_t,
                                     __set_tag_for_t<_LetTag>,
                                     __sndr2_fn<_Fn>::template __call,
                                     __all_non_dependent_t>::value;

    if constexpr (__all_non_dependent)
    {
      execution::__assert_valid_completion_signatures(get_completion_signatures<__sndr_t>());
    }
  }
  return transform_sender(__domain_t{}, __sndr_t{{{}, static_cast<_Fn&&>(__fn), static_cast<_Sndr&&>(__sndr)}});
}

template <class _LetTag>
template <class _Fn>
[[nodiscard]] _CCCL_NODEBUG_API constexpr auto __let_base_t<_LetTag>::operator()(_Fn __fn) const noexcept
{
  using __closure_t = typename _LetTag::template __closure_t<_Fn>;
  return __closure_t{{static_cast<_Fn&&>(__fn)}};
}

template <>
constexpr set_value_t __let_t::__set_tag<let_value_t>{};
template <>
constexpr set_error_t __let_t::__set_tag<let_error_t>{};
template <>
constexpr set_stopped_t __let_t::__set_tag<let_stopped_t>{};

template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<let_value_t::__sndr_t<_Sndr, _Fn>> = 3;
template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<let_error_t::__sndr_t<_Sndr, _Fn>> = 3;
template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<let_stopped_t::__sndr_t<_Sndr, _Fn>> = 3;

_CCCL_GLOBAL_CONSTANT auto let_value   = let_value_t{};
_CCCL_GLOBAL_CONSTANT auto let_error   = let_error_t{};
_CCCL_GLOBAL_CONSTANT auto let_stopped = let_stopped_t{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_LET_VALUE
