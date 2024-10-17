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

#include <cuda/std/atomic>

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/config.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/env.cuh>
#include <cuda/experimental/__async/exception.cuh>
#include <cuda/experimental/__async/lazy.cuh>
#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/stop_token.cuh>
#include <cuda/experimental/__async/tuple.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/utility.cuh>
#include <cuda/experimental/__async/variant.cuh>

#include <cuda/experimental/__async/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_NV_DIAG_SUPPRESS(expr_has_no_effect)
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-value")

namespace cuda::experimental::__async
{
// Forward declare the when_all tag type:
struct when_all_t;

// Some mechanics for computing a when_all sender's completion signatures:
namespace __when_all
{
template <class>
struct __env_t;

template <class, size_t>
struct __rcvr_t;

template <class, class, class>
struct __opstate_t;

using __tombstone = _ERROR<_WHERE(_IN_ALGORITHM, when_all_t), _WHAT(_SENDER_HAS_TOO_MANY_SUCCESS_COMPLETIONS)>;

// Use this to short-circuit the computation of whether all values and
// errors are nothrow decay-copyable.
template <class _Bool>
struct __all_nothrow_decay_copyable
{
  static_assert(_CUDA_VSTD::is_same_v<_Bool, __mtrue>);
  template <class... _Ts>
  using __f = __mbool<__nothrow_decay_copyable<_Ts...>>;
};

template <>
struct __all_nothrow_decay_copyable<__mfalse>
{
  template <class... _Ts>
  using __f = __mfalse;
};

//////////////////////////////////////////////////////////////////////////////////////
// This type is used to compute the completion signatures contributed by one of
// when_all's child senders. It tracks the completions, whether decay-copying the
// values and errors can throw, and also which of the when_all's value result
// datums this sender is responsible for setting.
//
// Leave this undefined:
template <class _NothrowVals, class _NothrowErrors, class _Offsets, class... _Sigs>
struct __completion_metadata;

//////////////////////////////////////////////////////////////////////////////////////
// Convert the metadata type into a completion signatures type by adding a
// set_error_t(exception_ptr) completion if decay-copying any of the values
// or errors is possibly throwing, and then removing duplicate completion
// signatures.
template <class>
struct __reduce_completions;

template <class... _What>
struct __reduce_completions<_ERROR<_What...>&>
{
  using type = __mpair<_ERROR<_What...>, __moffsets<>>;
};

template <class _ValsOK, class _ErrsOK, class _Offsets, class... _Sigs>
struct __reduce_completions<__completion_metadata<_ValsOK, _ErrsOK, _Offsets, _Sigs...>&>
{
  using type = __mpair< //
    __concat_completion_signatures<completion_signatures<_Sigs..., set_error_t(::std::exception_ptr)>>,
    _Offsets>;
};

template <class _Offsets, class... _Sigs>
struct __reduce_completions<__completion_metadata<__mtrue, __mtrue, _Offsets, _Sigs...>&>
{
  using type = __mpair<__concat_completion_signatures<completion_signatures<_Sigs...>>, _Offsets>;
};

template <class _Ty>
using __reduce_completions_t = __t<__reduce_completions<_Ty>>;

//////////////////////////////////////////////////////////////////////////////////////
// __append_completion
//
// We use a set of partial specialization of the __append_completion variable
// template to append the metadata from a single completion signature into a
// metadata struct, and we use a fold expression to append all _Ny completion
// signatures.
template <class _Metadata, class _Sig>
extern __undefined<_Metadata> __append_completion;

template <class _ValsOK, class _ErrsOK, class... _Sigs, class _Tag, class... _As>
extern __completion_metadata<_ValsOK,
                             __minvoke<__all_nothrow_decay_copyable<_ErrsOK>, _As...>,
                             __moffsets<>,
                             _Sigs...,
                             _Tag(__decay_t<_As>...)>&
  __append_completion<__completion_metadata<_ValsOK, _ErrsOK, __moffsets<>, _Sigs...>, _Tag(_As...)>;

// This overload is selected when we see the first set_value_t completion
// signature.
template <class _ValsOK, class _ErrsOK, class... _Sigs, class... _As>
extern __completion_metadata<
  __minvoke<__all_nothrow_decay_copyable<_ValsOK>, _As...>,
  _ErrsOK,
  __moffsets<>,
  set_value_t(__decay_t<_As>...), // Insert the value signature at the front
  _Sigs...>& __append_completion<__completion_metadata<_ValsOK, _ErrsOK, __moffsets<>, _Sigs...>, set_value_t(_As...)>;

// This overload is selected when we see the second set_value_t completion
// signature. Senders passed to when_all are only allowed one set_value
// completion.
template <class _ValsOK, class _ErrsOK, class... _Sigs, class... _As, class... _Bs>
extern __tombstone&
  __append_completion<__completion_metadata<_ValsOK, _ErrsOK, __moffsets<>, set_value_t(_As...), _Sigs...>,
                      set_value_t(_Bs...)>;

// This overload is selected when we see the second set_value_t completion
// signature. Senders passed to when_all are only allowed one set_value
// completion.
template <class _Sig>
extern __tombstone& __append_completion<__tombstone&, _Sig>;

// We use a fold expression over the bitwise OR operator to append all of the
// completion signatures from one child sender into a metadata struct.
template <class _Metadata, class _Sig>
auto operator|(_Metadata&, _Sig*) -> decltype(__append_completion<_Metadata, _Sig>);

// The initial value of the fold expression:
using __inner_fold_init = __completion_metadata<__mtrue, __mtrue, __moffsets<>>;

template <class... _Sigs>
using __collect_inner = //
  decltype((__declval<__inner_fold_init&>() | ... | static_cast<_Sigs*>(nullptr)));

//////////////////////////////////////////////////////////////////////////////////////
// __merge_metadata
//
// After computing a metadata struct for each child sender, all the metadata
// structs must be merged. We use a set of partial specialization of the
// __merge_metadata variable template to merge two metadata structs into one,
// and we use a fold expression to merge all _Ny into one.
template <class _Meta1, class _Meta2>
extern __undefined<_Meta1> __merge_metadata;

// This specialization causes an error to be propagated.
template <class _ValsOK, class _ErrsOK, class _Offsets, class... _LeftSigs, class... _What>
extern _ERROR<_What...>&
  __merge_metadata<__completion_metadata<_ValsOK, _ErrsOK, _Offsets, _LeftSigs...>, _ERROR<_What...>>;

// This overload is selected with the left and right metadata are both for senders
// that have no set_value completion signature.
template <class _LeftValsOK,
          class _LeftErrsOK,
          class _Offsets,
          class... _LeftSigs,
          class _RightValsOK,
          class _RightErrsOK,
          class... _RightSigs>
extern __completion_metadata<__mtrue, __mand<_LeftErrsOK, _RightErrsOK>, __moffsets<>, _LeftSigs..., _RightSigs...>&
  __merge_metadata<__completion_metadata<_LeftValsOK, _LeftErrsOK, _Offsets, _LeftSigs...>,
                   __completion_metadata<_RightValsOK, _RightErrsOK, __moffsets<>, _RightSigs...>>;

// The following two specializations are selected when one of the metadata
// structs is for a sender with no value completions. In that case, the
// when_all can never complete successfully, so drop the other set_value
// completion signature.
template <class _LeftValsOK,
          class _LeftErrsOK,
          class _Offsets,
          class... _As,
          class... _LeftSigs,
          class _RightValsOK,
          class _RightErrsOK,
          class... _RightSigs>
extern __completion_metadata<__mtrue, // There will be no value completion, so values need not be copied.
                             __mand<_LeftErrsOK, _RightErrsOK>,
                             __moffsets<>,
                             _LeftSigs...,
                             _RightSigs...>&
  __merge_metadata<__completion_metadata<_LeftValsOK, _LeftErrsOK, _Offsets, set_value_t(_As...), _LeftSigs...>,
                   __completion_metadata<_RightValsOK, _RightErrsOK, __moffsets<>, _RightSigs...>>;

template <class _LeftValsOK,
          class _LeftErrsOK,
          class _Offsets,
          class... _LeftSigs,
          class _RightValsOK,
          class _RightErrsOK,
          class... _As,
          class... _RightSigs>
extern __completion_metadata<__mtrue, // There will be no value completion, so values need not be copied.
                             __mand<_LeftErrsOK, _RightErrsOK>,
                             __moffsets<>,
                             _LeftSigs...,
                             _RightSigs...>&
  __merge_metadata<__completion_metadata<_LeftValsOK, _LeftErrsOK, _Offsets, _LeftSigs...>,
                   __completion_metadata<_RightValsOK, _RightErrsOK, __moffsets<>, set_value_t(_As...), _RightSigs...>>;

template <size_t... _Offsets>
_CCCL_INLINE_VAR constexpr size_t __last_offset = (0, ..., _Offsets);

template <size_t _Count, size_t... _Offsets>
using __append_offset = __moffsets<_Offsets..., _Count + __last_offset<_Offsets...>>;

// This overload is selected when both metadata structs are for senders with
// a single value completion. Concatenate the value types.
template <class _LeftValsOK,
          class _LeftErrsOK,
          size_t... _Offsets,
          class... _As,
          class... _LeftSigs,
          class _RightValsOK,
          class _RightErrsOK,
          class... _Bs,
          class... _RightSigs>
extern __completion_metadata<__mand<_LeftValsOK, _RightValsOK>,
                             __mand<_LeftErrsOK, _RightErrsOK>,
                             __append_offset<sizeof...(_Bs), _Offsets...>,
                             set_value_t(_As..., _Bs...), // Concatenate the value types.
                             _LeftSigs...,
                             _RightSigs...>&
  __merge_metadata<
    __completion_metadata<_LeftValsOK, _LeftErrsOK, __moffsets<_Offsets...>, set_value_t(_As...), _LeftSigs...>,
    __completion_metadata<_RightValsOK, _RightErrsOK, __moffsets<>, set_value_t(_Bs...), _RightSigs...>>;

template <class... _What, class _Other>
extern _ERROR<_What...>& __merge_metadata<_ERROR<_What...>, _Other>;

// We use a fold expression over the bitwise AND operator to merge all the
// completion metadata structs from the child senders into a single metadata
// struct.
template <class _Meta1, class _Meta2>
auto operator&(_Meta1&, _Meta2&) -> decltype(__merge_metadata<_Meta1, _Meta2>);

// The initial value for the fold.
using __outer_fold_init = __completion_metadata<__mtrue, __mtrue, __moffsets<0ul>, set_value_t(), set_stopped_t()>;

template <class... _Sigs>
using __collect_outer = //
  __reduce_completions_t<decltype((__declval<__outer_fold_init&>() & ... & __declval<_Sigs>()))>;

// Extract the first template parameter of the __state_t specialization.
// The first template parameter is the receiver type.
template <class _State>
using __rcvr_from_state_t = __mapply<__mpoly_q<__mfront>, _State>;

/// The receivers connected to the when_all's sub-operations expose this as
/// their environment. Its `get_stop_token` query returns the token from
/// when_all's stop source. All other queries are forwarded to the outer
/// receiver's environment.
template <class _StateZip>
struct __env_t
{
  using __state_t = __unzip<_StateZip>;
  using __rcvr_t  = __rcvr_from_state_t<__state_t>;

  __state_t& __state_;

  _CCCL_HOST_DEVICE inplace_stop_token __query(get_stop_token_t) const noexcept
  {
    return __state_.__stop_token_;
  }

  template <class _Tag>
  _CCCL_HOST_DEVICE auto query(_Tag) const noexcept -> __query_result_t<_Tag, env_of_t<__rcvr_t>>
  {
    return __async::get_env(__state_.__rcvr_).__query(_Tag());
  }
};

template <class _StateZip, size_t _Index>
struct __rcvr_t
{
  using receiver_concept = receiver_t;
  using __state_t        = __unzip<_StateZip>;

  __state_t& __state_;

  template <class... _Ts>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_value(_Ts&&... __ts) noexcept
  {
    constexpr auto idx = __mmake_indices<sizeof...(_Ts)>();
    __state_.template __set_value<_Index>(idx, static_cast<_Ts&&>(__ts)...);
    __state_.__arrive();
  }

  template <class _Error>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_error(_Error&& __error) noexcept
  {
    __state_.__set_error(static_cast<_Error&&>(__error));
    __state_.__arrive();
  }

  _CCCL_HOST_DEVICE void set_stopped() noexcept
  {
    __state_.__set_stopped();
    __state_.__arrive();
  }

  _CCCL_HOST_DEVICE auto get_env() const noexcept -> __env_t<_StateZip>
  {
    return {__state_};
  }
};

template <class _CvSndr, size_t _Idx, class _StateZip>
using __inner_completions_ = //
  __mapply_q<__collect_inner, completion_signatures_of_t<_CvSndr, __rcvr_t<_StateZip, _Idx>>>;

template <class _CvSndr, size_t _Idx, class _StateZip>
using __inner_completions = //
  __midentity_or_error_with< //
    __inner_completions_<_CvSndr, _Idx, _StateZip>, //
    _WITH_SENDER(_CvSndr)>;

enum __estate_t : int
{
  __started,
  __error,
  __stopped
};

/// @brief The data stored in the operation state and refered to
/// by the receiver.
/// @tparam _Rcvr The receiver connected to the when_all sender.
/// @tparam _CvFn A metafunction to apply cv- and ref-qualifiers to the senders
/// @tparam _Sndrs A tuple of the when_all sender's child senders.
template <class _Rcvr, class _CvFn, class _Sndrs>
struct __state_t;

template <class _Rcvr, class _CvFn, size_t... _Idx, class... _Sndrs>
struct __state_t<_Rcvr, _CvFn, __tupl<__mindices<_Idx...>, _Sndrs...>>
{
  using __completions_offsets_pair_t = //
    __collect_outer< //
      __inner_completions<__minvoke1<_CvFn, _Sndrs>, _Idx, __zip<__state_t>>...>;
  using __completions_t = __mfirst<__completions_offsets_pair_t>;
  using __indices_t     = __mindices<_Idx...>;
  using __offsets_t     = __msecond<__completions_offsets_pair_t>;
  using __values_t      = __value_types<__completions_t, __lazy_tuple, __mpoly<__msingle_or<__nil>>::__f>;
  using __errors_t      = __error_types<__completions_t, __variant>;

  using __stop_tok_t      = stop_token_of_t<env_of_t<_Rcvr>>;
  using __stop_callback_t = stop_callback_for_t<__stop_tok_t, __on_stop_request>;

  _CCCL_HOST_DEVICE explicit __state_t(_Rcvr __rcvr, size_t __count)
      : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
      , __count_{__count}
      , __stop_source_{}
      , __stop_token_{__stop_source_.get_token()}
      , __state_{__started}
      , __errors_{}
      , __values_{}
      , __on_stop_{}
  {}

  template <size_t _Index, size_t... _Offsets>
  static constexpr size_t __offset_for(__moffsets<_Offsets...>*) noexcept
  {
    constexpr size_t __offsets[] = {_Offsets..., 0};
    return __offsets[_Index];
  }

  template <size_t _Index, size_t... _Jdx, class... _Ts>
  _CCCL_HOST_DEVICE void __set_value(__mindices<_Jdx...>, [[maybe_unused]] _Ts&&... __ts) noexcept
  {
    [[maybe_unused]] constexpr size_t _Offset = __offset_for<_Index>(static_cast<__offsets_t*>(nullptr));
    if constexpr (!_CUDA_VSTD::is_same_v<__values_t, __nil>)
    {
      if constexpr (__nothrow_decay_copyable<_Ts...>)
      {
        (__values_.template __emplace<_Jdx + _Offset>(static_cast<_Ts&&>(__ts)), ...);
      }
      else
      {
        _CUDAX_TRY( //
          ({ //
            (__values_.template __emplace<_Jdx + _Offset>(static_cast<_Ts&&>(__ts)), ...);
          }),
          _CUDAX_CATCH(...)( //
            { //
              __set_error(::std::current_exception());
            }))
      }
    }
  }

  template <class _Error>
  _CCCL_HOST_DEVICE void __set_error(_Error&& __err) noexcept
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
          ({ //
            __errors_.template __emplace<__decay_t<_Error>>(static_cast<_Error&&>(__err));
          }),
          _CUDAX_CATCH(...)( //
            { //
              __errors_.template __emplace<::std::exception_ptr>(::std::current_exception());
            }))
      }
    }
  }

  _CCCL_HOST_DEVICE void __set_stopped() noexcept
  {
    _CUDA_VSTD::underlying_type_t<__estate_t> __expected = __started;
    // Transition to the "stopped" state if and only if we're in the
    // "started" state. (If this fails, it's because we're in an
    // error state, which trumps cancellation.)
    if (__state_.compare_exchange_strong(__expected, static_cast<_CUDA_VSTD::underlying_type_t<__estate_t>>(__stopped)))
    {
      __stop_source_.request_stop();
    }
  }

  _CCCL_HOST_DEVICE void __arrive() noexcept
  {
    if (0 == --__count_)
    {
      __complete();
    }
  }

  _CCCL_HOST_DEVICE void __complete() noexcept
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
  __values_t __values_;
  __lazy<__stop_callback_t> __on_stop_;
};

/// The operation state for when_all
template <class _Rcvr, class _CvFn, size_t... _Idx, class... _Sndrs>
struct __opstate_t<_Rcvr, _CvFn, __tupl<__mindices<_Idx...>, _Sndrs...>>
{
  using operation_state_concept = operation_state_t;
  using __sndrs_t               = __minvoke<_CvFn, __tuple<_Sndrs...>>;
  using __state_t               = __when_all::__state_t<_Rcvr, _CvFn, __tupl<__mindices<_Idx...>, _Sndrs...>>;

  using completion_signatures = typename __state_t::__completions_t;
  using __offsets_t           = typename __state_t::__offsets_t;

  // This function object is used to connect all the sub-operations with
  // receivers, each of which knows which elements in the values tuple it
  // is responsible for setting.
  struct __connect_subs_fn
  {
    template <class... _CvSndrs>
    _CCCL_HOST_DEVICE auto operator()(__state_t& __state, _CvSndrs&&... __sndrs_) const
    {
      using __state_ref_t = __zip<__state_t>;
      if constexpr (_CUDA_VSTD::is_same_v<__offsets_t, __moffsets<>>)
      {
        // When there are no offsets, the when_all sender has no value
        // completions. All child senders can be connected to receivers
        // of the same type.
        return __tupl{__async::connect(static_cast<_CvSndrs&&>(__sndrs_), __rcvr_t<__state_ref_t, 0>{__state})...};
      }
      else
      {
        // The offsets are used to determine which elements in the values
        // tuple each receiver is responsible for setting.
        return __tupl{__async::connect(static_cast<_CvSndrs&&>(__sndrs_), __rcvr_t<__state_ref_t, _Idx>{__state})...};
      }
    }
  };

  // This is a tuple of operation states for the sub-operations.
  using __sub_opstates_t = __apply_result_t<__connect_subs_fn, __sndrs_t, __state_t&>;

  __state_t __state_;
  __sub_opstates_t __sub_ops_;

  /// Initialize the data member, connect all the sub-operations and
  /// save the resulting operation states in __sub_ops_.
  _CCCL_HOST_DEVICE __opstate_t(__sndrs_t&& __sndrs_, _Rcvr __rcvr)
      : __state_{static_cast<_Rcvr&&>(__rcvr), sizeof...(_Sndrs)}
      , __sub_ops_{__sndrs_.__apply(__connect_subs_fn(), static_cast<__sndrs_t&&>(__sndrs_), __state_)}
  {}

  _CUDAX_IMMOVABLE(__opstate_t);

  /// Start all the sub-operations.
  _CCCL_HOST_DEVICE void start() & noexcept
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
      __sub_ops_.__for_each(__async::start, __sub_ops_);

      // If there are no sub-operations, we're done.
      if constexpr (sizeof...(_Sndrs) == 0)
      {
        __state_.__complete();
      }
    }
  }
};

template <class... _Sndrs>
struct __sndr_t;
} // namespace __when_all

struct when_all_t
{
  template <class... _Sndrs>
  _CCCL_HOST_DEVICE __when_all::__sndr_t<_Sndrs...> operator()(_Sndrs... __sndrs_) const;
};

// The sender for when_all
template <class... _Sndrs>
struct __when_all::__sndr_t
{
  using sender_concept = sender_t;
  using __sndrs_t      = __tuple<_Sndrs...>;

  _CCCL_NO_UNIQUE_ADDRESS when_all_t __tag_;
  _CCCL_NO_UNIQUE_ADDRESS __ignore __ignore1_;
  __sndrs_t __sndrs_;

  template <class _Rcvr>
  _CCCL_HOST_DEVICE auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, __cp, __sndrs_t>
  {
    return __opstate_t<_Rcvr, __cp, __sndrs_t>(static_cast<__sndrs_t&&>(__sndrs_), static_cast<_Rcvr&&>(__rcvr));
  }

  template <class _Rcvr>
  _CCCL_HOST_DEVICE auto connect(_Rcvr __rcvr) const& //
    -> __opstate_t<_Rcvr, __cpclr, __sndrs_t>
  {
    return __opstate_t<_Rcvr, __cpclr, __sndrs_t>(__sndrs_, static_cast<_Rcvr&&>(__rcvr));
  }
};

template <class... _Sndrs>
_CCCL_HOST_DEVICE __when_all::__sndr_t<_Sndrs...> when_all_t::operator()(_Sndrs... __sndrs_) const
{
  // If the incoming sender is non-dependent, we can check the completion
  // signatures of the composed sender immediately.
  if constexpr ((__is_non_dependent_sender<_Sndrs> && ...))
  {
    using __completions = completion_signatures_of_t<__when_all::__sndr_t<_Sndrs...>>;
    static_assert(__is_completion_signatures<__completions>);
  }
  return __when_all::__sndr_t<_Sndrs...>{{}, {}, {static_cast<_Sndrs&&>(__sndrs_)...}};
}

_CCCL_GLOBAL_CONSTANT when_all_t when_all{};

} // namespace cuda::experimental::__async

_CCCL_NV_DIAG_DEFAULT(expr_has_no_effect)
_CCCL_DIAG_POP

#include <cuda/experimental/__async/epilogue.cuh>

#endif
