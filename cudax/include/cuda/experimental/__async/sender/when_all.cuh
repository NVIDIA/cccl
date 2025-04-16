//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_WHEN_ALL
#define __CUDAX_ASYNC_DETAIL_WHEN_ALL

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/atomic>

#include <cuda/experimental/__async/sender/completion_signatures.cuh>
#include <cuda/experimental/__async/sender/concepts.cuh>
#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/env.cuh>
#include <cuda/experimental/__async/sender/exception.cuh>
#include <cuda/experimental/__async/sender/lazy.cuh>
#include <cuda/experimental/__async/sender/meta.cuh>
#include <cuda/experimental/__async/sender/stop_token.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__async/sender/type_traits.cuh>
#include <cuda/experimental/__async/sender/utility.cuh>
#include <cuda/experimental/__async/sender/variant.cuh>
#include <cuda/experimental/__async/sender/visit.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT when_all_t
{
  template <class... _Sndrs>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

private:
  // Extract the first template parameter of the __state_t specialization.
  // The first template parameter is the receiver type.
  template <class _State>
  using __rcvr_from_state_t = _CUDA_VSTD::__type_apply<_CUDA_VSTD::__detail::__type_at_fn<0>, _State>;

  // Returns the completion signatures of a child sender. Throws an exception if
  // the child sender has more than one set_value completion signature.
  template <class _Child, class... _Env>
  _CUDAX_API static constexpr auto __child_completions();

  // Merges the completion signatures of the child senders into a single set of
  // completion signatures for the when_all sender.
  template <class... _Completions>
  _CUDAX_API static constexpr auto __merge_completions(_Completions...);

  /// The receivers connected to the when_all's sub-operations expose this as
  /// their environment. Its `get_stop_token` query returns the token from
  /// when_all's stop source. All other queries are forwarded to the outer
  /// receiver's environment.
  template <class _StateZip>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    using __state_t = __unzip<_StateZip>;
    using __rcvr_t  = __rcvr_from_state_t<__state_t>;

    __state_t& __state_;

    _CUDAX_API inplace_stop_token query(get_stop_token_t) const noexcept
    {
      return __state_.__stop_token_;
    }

    // TODO: only forward the "forwarding" queries
    template <class _Tag>
    _CUDAX_API auto query(_Tag) const noexcept -> __query_result_t<_Tag, env_of_t<__rcvr_t>>
    {
      return __async::get_env(__state_.__rcvr_).query(_Tag());
    }
  };

  template <class _StateZip, size_t _Index>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;
    using __state_t        = __unzip<_StateZip>;

    __state_t& __state_;

    template <class... _Ts>
    _CUDAX_TRIVIAL_API void set_value(_Ts&&... __ts) noexcept
    {
      constexpr _CUDA_VSTD::index_sequence_for<_Ts...>* idx = nullptr;
      __state_.template __set_value<_Index>(idx, static_cast<_Ts&&>(__ts)...);
      __state_.__arrive();
    }

    template <class _Error>
    _CUDAX_TRIVIAL_API void set_error(_Error&& __error) noexcept
    {
      __state_.__set_error(static_cast<_Error&&>(__error));
      __state_.__arrive();
    }

    _CUDAX_API void set_stopped() noexcept
    {
      __state_.__set_stopped();
      __state_.__arrive();
    }

    _CUDAX_API auto get_env() const noexcept -> __env_t<_StateZip>
    {
      return {__state_};
    }
  };

  enum __estate_t : int
  {
    __started,
    __error,
    __stopped
  };

  /// @brief The data stored in the operation state and referred to
  /// by the receiver.
  /// @tparam _Rcvr The receiver connected to the when_all sender.
  /// @tparam _CvFn A metafunction to apply cv- and ref-qualifiers to the senders
  /// @tparam _Sndrs A tuple of the when_all sender's child senders.
  template <class _Rcvr, class _CvFn, class _Sndrs>
  struct __state_t;

  template <class _Rcvr, class _CvFn, class _Idx, class _Ign0, class _Ign1, class... _Sndrs>
  struct __state_t<_Rcvr, _CvFn, __tupl<_Idx, _Ign0, _Ign1, _Sndrs...>>
  {
    using __env_t     = when_all_t::__env_t<__zip<__state_t>>;
    using __sndr_t    = when_all_t::__sndr_t<_Sndrs...>;
    using __cv_sndr_t = _CUDA_VSTD::__type_call1<_CvFn, __sndr_t>;

    static constexpr auto __completions_and_offsets =
      __sndr_t::template __get_completions_and_offsets<__cv_sndr_t, __env_t>();

    using __completions_t   = decltype(__completions_and_offsets.first);
    using __values_t        = __value_types<__completions_t, __lazy_tuple, __type_self_or<__nil>::__call>;
    using __errors_t        = __error_types<__completions_t, __variant>;
    using __stop_tok_t      = stop_token_of_t<env_of_t<_Rcvr>>;
    using __stop_callback_t = stop_callback_for_t<__stop_tok_t, __on_stop_request>;

    _CUDAX_API explicit __state_t(_Rcvr __rcvr, size_t __count)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __count_{__count}
        , __stop_source_{}
        , __stop_token_{__stop_source_.get_token()}
        , __state_{__started}
        , __errors_{}
        , __values_{}
        , __on_stop_{}
    {}

    template <size_t _Index, size_t... _Jdx, class... _Ts>
    _CUDAX_API void __set_value(_CUDA_VSTD::index_sequence<_Jdx...>*, [[maybe_unused]] _Ts&&... __ts) noexcept
    {
      if constexpr (!_CUDA_VSTD::is_same_v<__values_t, __nil>)
      {
        constexpr size_t _Offset = __completions_and_offsets.second[_Index];
        if constexpr (__nothrow_decay_copyable<_Ts...>)
        {
          (__values_.template __emplace<_Jdx + _Offset>(static_cast<_Ts&&>(__ts)), ...);
        }
        else
        {
          _CUDAX_TRY( //
            ({        //
              (__values_.template __emplace<_Jdx + _Offset>(static_cast<_Ts&&>(__ts)), ...);
            }),
            _CUDAX_CATCH(...) //
            ({                //
              __set_error(::std::current_exception());
            }) //
          )
        }
      }
    }

    template <class _Error>
    _CUDAX_API void __set_error(_Error&& __err) noexcept
    {
      // TODO: Use weaker memory orders
      if (__error != __state_.exchange(__error))
      {
        __stop_source_.request_stop();
        // We won the race, free to write the error into the operation state
        // without worry.
        if constexpr (__nothrow_decay_copyable<_Error>)
        {
          __errors_.template __emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
        }
        else
        {
          _CUDAX_TRY( //
            ({        //
              __errors_.template __emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
            }),
            _CUDAX_CATCH(...) //
            ({                //
              __errors_.template __emplace<::std::exception_ptr>(::std::current_exception());
            }) //
          )
        }
      }
    }

    _CUDAX_API void __set_stopped() noexcept
    {
      _CUDA_VSTD::underlying_type_t<__estate_t> __expected = __started;
      // Transition to the "stopped" state if and only if we're in the
      // "started" state. (If this fails, it's because we're in an
      // error state, which trumps cancellation.)
      if (__state_.compare_exchange_strong(
            __expected, static_cast<_CUDA_VSTD::underlying_type_t<__estate_t>>(__stopped)))
      {
        __stop_source_.request_stop();
      }
    }

    _CUDAX_API void __arrive() noexcept
    {
      if (0 == --__count_)
      {
        __complete();
      }
    }

    _CUDAX_API void __complete() noexcept
    {
      // Stop callback is no longer needed. Destroy it.
      __on_stop_.destroy();
      // All child operations have completed and arrived at the barrier.
      switch (__state_.load(_CUDA_VSTD::memory_order_relaxed))
      {
        case __started:
          if constexpr (!_CUDA_VSTD::is_same_v<__values_t, __nil>)
          {
            // All child operations completed successfully:
            __values_.__apply(__async::set_value, static_cast<__values_t&&>(__values_), static_cast<_Rcvr&&>(__rcvr_));
          }
          break;
        case __error:
          // One or more child operations completed with an error:
          __errors_.__visit(__async::set_error, static_cast<__errors_t&&>(__errors_), static_cast<_Rcvr&&>(__rcvr_));
          break;
        case __stopped:
          __async::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
          break;
        default:;
      }
    }

    _Rcvr __rcvr_;
    _CUDA_VSTD::atomic<size_t> __count_;
    inplace_stop_source __stop_source_;
    inplace_stop_token __stop_token_;
    _CUDA_VSTD::atomic<_CUDA_VSTD::underlying_type_t<__estate_t>> __state_;
    __errors_t __errors_;
    // _CCCL_NO_UNIQUE_ADDRESS // gcc doesn't like this
    __values_t __values_;
    __lazy<__stop_callback_t> __on_stop_;
  };

  struct __start_all
  {
    template <class... _Ops>
    _CUDAX_TRIVIAL_API void operator()(_Ops&... __ops) const noexcept
    {
      (__async::start(__ops), ...);
    }
  };

  /// The operation state for when_all
  template <class, class, class>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t;

  template <class _Rcvr, class _CvFn, size_t... _Idx, class _Ign0, class _Ign1, class... _Sndrs>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT
  __opstate_t<_Rcvr, _CvFn, __tupl<_CUDA_VSTD::index_sequence<0, 1, _Idx...>, _Ign0, _Ign1, _Sndrs...>>
  {
    using operation_state_concept = operation_state_t;
    using __sndrs_t               = _CUDA_VSTD::__type_call<_CvFn, __tuple<_Ign0, _Ign1, _Sndrs...>>;
    using __state_t               = when_all_t::__state_t<_Rcvr, _CvFn, __tuple<_Ign0, _Ign1, _Sndrs...>>;

    // This function object is used to connect all the sub-operations with
    // receivers, each of which knows which elements in the values tuple it
    // is responsible for setting.
    struct __connect_subs_fn
    {
      template <class... _CvSndrs>
      _CUDAX_API auto operator()(__state_t& __state, __ignore, __ignore, _CvSndrs&&... __sndrs_) const
      {
        using __state_ref_t = __zip<__state_t>;
        // When there are no offsets, the when_all sender has no value
        // completions. All child senders can be connected to receivers
        // of the same type, saving template instantiations.
        [[maybe_unused]] constexpr bool __no_values =
          _CUDA_VSTD::is_same_v<decltype(__state_t::__completions_and_offsets.second), __nil>;
        // The offsets are used to determine which elements in the values
        // tuple each receiver is responsible for setting.
        return __tupl{__async::connect(
          static_cast<_CvSndrs&&>(__sndrs_), __rcvr_t<__state_ref_t, __no_values ? 0 : _Idx - 2>{__state})...};
      }
    };

    // This is a tuple of operation states for the sub-operations.
    using __sub_opstates_t = __apply_result_t<__connect_subs_fn, __sndrs_t, __state_t&>;

    __state_t __state_;
    __sub_opstates_t __sub_ops_;

    /// Initialize the data member, connect all the sub-operations and
    /// save the resulting operation states in __sub_ops_.
    _CUDAX_API __opstate_t(__sndrs_t&& __sndrs_, _Rcvr __rcvr)
        : __state_{static_cast<_Rcvr&&>(__rcvr), sizeof...(_Sndrs)}
        , __sub_ops_{__sndrs_.__apply(__connect_subs_fn(), static_cast<__sndrs_t&&>(__sndrs_), __state_)}
    {}

    _CUDAX_IMMOVABLE(__opstate_t);

    /// Start all the sub-operations.
    _CUDAX_API void start() & noexcept
    {
      // register stop callback:
      __state_.__on_stop_.construct(
        get_stop_token(__async::get_env(__state_.__rcvr_)), __on_stop_request{__state_.__stop_source_});

      if (__state_.__stop_source_.stop_requested())
      {
        // Manually clean up the stop callback. We won't be starting the
        // sub-operations, so they won't complete and clean up for us.
        __state_.__on_stop_.destroy();

        // Stop has already been requested. Don't bother starting the child
        // operations.
        __async::set_stopped(static_cast<_Rcvr&&>(__state_.__rcvr_));
      }
      else
      {
        // Start all the sub-operations.
        __sub_ops_.__apply(__start_all{}, __sub_ops_);

        // If there are no sub-operations, we're done.
        if constexpr (sizeof...(_Sndrs) == 0)
        {
          __state_.__complete();
        }
      }
    }
  };

  template <class... _Ts>
  using __decay_all = _CUDA_VSTD::__type_list<_CUDA_VSTD::decay_t<_Ts>...>;

public:
  template <class... _Sndrs>
  _CUDAX_API auto operator()(_Sndrs... __sndrs) const -> __sndr_t<_Sndrs...>;
};

template <class _Child, class... _Env>
_CUDAX_API constexpr auto when_all_t::__child_completions()
{
  using __env_t = prop<get_stop_token_t, inplace_stop_token>;
  _CUDAX_LET_COMPLETIONS(auto(__completions) = get_completion_signatures<_Child, env<__env_t, _FWD_ENV_T<_Env>>...>())
  {
    if constexpr (__completions.count(set_value) > 1)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, when_all_t),
                                          _WHAT(_SENDER_HAS_TOO_MANY_SUCCESS_COMPLETIONS),
                                          _WITH_SENDER(_Child)>();
    }
    else
    {
      return __completions;
    }
  }
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-value")

template <class... _Completions>
_CUDAX_API constexpr auto when_all_t::__merge_completions(_Completions... __cs)
{
  // Use _CUDAX_LET_COMPLETIONS to ensure all completions are valid:
  _CUDAX_LET_COMPLETIONS(auto(__tmp) = (completion_signatures{}, ..., __cs)) // NB: uses overloaded comma operator
  {
    _CUDA_VSTD::ignore           = __tmp; // silence unused variable warning
    auto __non_value_completions = concat_completion_signatures(
      completion_signatures<set_stopped_t()>(),
      transform_completion_signatures(__cs, __swallow_transform(), __decay_transform<set_error_t>())...);

    if constexpr (((0 == __cs.count(set_value)) || ...))
    {
      // at least one child sender has no value completions at all, so the
      // when_all will never complete with set_value. return just the error and
      // stopped completions.
      return __pair{__non_value_completions, __nil{}};
    }
    else
    {
      std::array<size_t, sizeof...(_Completions)> __offsets = {
        __value_types<_Completions, _CUDA_VSTD::__type_list, _CUDA_VSTD::__type_list_size>::value...};
      (void) _CUDA_VSTD::exclusive_scan(__offsets.begin(), __offsets.end(), __offsets.begin(), std::size_t(0));

      // All child senders have exactly one value completion signature, each of
      // which may have multiple arguments. Concatenate all the arguments into a
      // single set_value_t completion signature.
      using __values_t = _CUDA_VSTD::__type_call<         //
        __type_concat_into<__type_function<set_value_t>>, //
        __value_types<_Completions, __decay_all, _CUDA_VSTD::__type_self_t>...>;
      // Add the value completion to the error and stopped completions.
      auto __local = __non_value_completions + completion_signatures<__values_t>();
      // Check if any of the values or errors are not nothrow decay-copyable.
      constexpr bool __all_nothrow_decay_copyable =
        (__value_types<_Completions, __nothrow_decay_copyable_t, __identity_t>::value && ...);
      return __pair{__local + __eptr_completion_if<!__all_nothrow_decay_copyable>(), __offsets};
    }
  }

  _CCCL_UNREACHABLE();
}

_CCCL_DIAG_POP

// The sender for when_all
template <class... _Sndrs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT when_all_t::__sndr_t : __tuple<when_all_t, __ignore, _Sndrs...>
{
  using sender_concept = sender_t;
  using __sndrs_t      = __tuple<when_all_t, __ignore, _Sndrs...>;

  template <class _Self, class... _Env>
  _CUDAX_API static constexpr auto __get_completions_and_offsets()
  {
    return __merge_completions(__child_completions<__copy_cvref_t<_Self, _Sndrs>, _Env...>()...);
  }

  template <class _Self, class... _Env>
  _CUDAX_API static constexpr auto get_completion_signatures()
  {
    return __get_completions_and_offsets<_Self, _Env...>().first;
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, __cp, __sndrs_t>
  {
    return __opstate_t<_Rcvr, __cp, __sndrs_t>(static_cast<__sndrs_t&&>(*this), static_cast<_Rcvr&&>(__rcvr));
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Rcvr, __cpclr, __sndrs_t>
  {
    return __opstate_t<_Rcvr, __cpclr, __sndrs_t>(static_cast<__sndrs_t const&>(*this), static_cast<_Rcvr&&>(__rcvr));
  }
};

template <class... _Sndrs>
_CUDAX_API auto when_all_t::operator()(_Sndrs... __sndrs) const -> __sndr_t<_Sndrs...>
{
  // If the incoming senders are non-dependent, we can check the completion
  // signatures of the composed sender immediately.
  if constexpr (((!dependent_sender<_Sndrs>) && ...))
  {
    using __completions = completion_signatures_of_t<__sndr_t<_Sndrs...>>;
    static_assert(__valid_completion_signatures<__completions>);
  }
  return __sndr_t<_Sndrs...>{{{}, {}, static_cast<_Sndrs&&>(__sndrs)...}};
}

template <class... _Sndrs>
inline constexpr size_t structured_binding_size<when_all_t::__sndr_t<_Sndrs...>> = sizeof...(_Sndrs) + 2;

_CCCL_GLOBAL_CONSTANT when_all_t when_all{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
