//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/atomic>

#include "completion_signatures.cuh"
#include "config.cuh"
#include "cpos.cuh"
#include "env.cuh"
#include "exception.cuh"
#include "lazy.cuh"
#include "meta.cuh"
#include "stop_token.cuh"
#include "tuple.cuh"
#include "type_traits.cuh"
#include "utility.cuh"
#include "variant.cuh"

// This must be the last #include
#include "prologue.cuh"

_CCCL_DIAG_PUSH
_CCCL_NV_DIAG_SUPPRESS(expr_has_no_effect)
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-value")

namespace cuda::experimental::__async
{
// Forward declare the when_all tag type:
struct when_all_t;

// Some mechanics for computing a when_all sender's completion signatures:
namespace _when_all
{
template <class>
struct _env_t;

template <class, size_t>
struct _rcvr_t;

template <class, class, class>
struct _opstate_t;

using _tombstone = ERROR<WHERE(IN_ALGORITHM, when_all_t), WHAT(SENDER_HAS_TOO_MANY_SUCCESS_COMPLETIONS)>;

// Use this to short-circuit the computation of whether all values and
// errors are nothrow decay-copyable.
template <class Bool>
struct _all_nothrow_decay_copyable
{
  static_assert(_CUDA_VSTD::is_same_v<Bool, _mtrue>);
  template <class... Ts>
  using _f = _mbool<_nothrow_decay_copyable<Ts...>>;
};

template <>
struct _all_nothrow_decay_copyable<_mfalse>
{
  template <class... Ts>
  using _f = _mfalse;
};

//////////////////////////////////////////////////////////////////////////////////////
// This type is used to compute the completion signatures contributed by one of
// when_all's child senders. It tracks the completions, whether decay-copying the
// values and errors can throw, and also which of the when_all's value result
// datums this sender is responsible for setting.
//
// Leave this undefined:
template <class NothrowVals, class NothrowErrors, class Offsets, class... Sigs>
struct _completion_metadata;

//////////////////////////////////////////////////////////////////////////////////////
// Convert the metadata type into a completion signatures type by adding a
// set_error_t(exception_ptr) completion if decay-copying any of the values
// or errors is possibly throwing, and then removing duplicate completion
// signatures.
template <class>
struct _reduce_completions;

template <class... What>
struct _reduce_completions<ERROR<What...>&>
{
  using type = _mpair<ERROR<What...>, _moffsets<>>;
};

template <class ValsOK, class ErrsOK, class Offsets, class... Sigs>
struct _reduce_completions<_completion_metadata<ValsOK, ErrsOK, Offsets, Sigs...>&>
{
  using type = _mpair< //
    _concat_completion_signatures<completion_signatures<Sigs..., set_error_t(::std::exception_ptr)>>,
    Offsets>;
};

template <class Offsets, class... Sigs>
struct _reduce_completions<_completion_metadata<_mtrue, _mtrue, Offsets, Sigs...>&>
{
  using type = _mpair<_concat_completion_signatures<completion_signatures<Sigs...>>, Offsets>;
};

template <class Ty>
using _reduce_completions_t = _t<_reduce_completions<Ty>>;

//////////////////////////////////////////////////////////////////////////////////////
// _append_completion
//
// We use a set of partial specialization of the _append_completion variable
// template to append the metadata from a single completion signature into a
// metadata struct, and we use a fold expression to append all N completion
// signatures.
template <class Metadata, class Sig>
extern _undefined<Metadata> _append_completion;

template <class ValsOK, class ErrsOK, class... Sigs, class Tag, class... As>
extern _completion_metadata<
  ValsOK,
  _minvoke<_all_nothrow_decay_copyable<ErrsOK>, As...>,
  _moffsets<>,
  Sigs...,
  Tag(_decay_t<As>...)>& _append_completion<_completion_metadata<ValsOK, ErrsOK, _moffsets<>, Sigs...>, Tag(As...)>;

// This overload is selected when we see the first set_value_t completion
// signature.
template <class ValsOK, class ErrsOK, class... Sigs, class... As>
extern _completion_metadata<
  _minvoke<_all_nothrow_decay_copyable<ValsOK>, As...>,
  ErrsOK,
  _moffsets<>,
  set_value_t(_decay_t<As>...), // Insert the value signature at the front
  Sigs...>& _append_completion<_completion_metadata<ValsOK, ErrsOK, _moffsets<>, Sigs...>, set_value_t(As...)>;

// This overload is selected when we see the second set_value_t completion
// signature. Senders passed to when_all are only allowed one set_value
// completion.
template <class ValsOK, class ErrsOK, class... Sigs, class... As, class... Bs>
extern _tombstone&
  _append_completion<_completion_metadata<ValsOK, ErrsOK, _moffsets<>, set_value_t(As...), Sigs...>, set_value_t(Bs...)>;

// This overload is selected when we see the second set_value_t completion
// signature. Senders passed to when_all are only allowed one set_value
// completion.
template <class Sig>
extern _tombstone& _append_completion<_tombstone&, Sig>;

// We use a fold expression over the bitwise OR operator to append all of the
// completion signatures from one child sender into a metadata struct.
template <class Metadata, class Sig>
auto operator|(Metadata&, Sig*) -> decltype(_append_completion<Metadata, Sig>);

// The initial value of the fold expression:
using _inner_fold_init = _completion_metadata<_mtrue, _mtrue, _moffsets<>>;

template <class... Sigs>
using _collect_inner = //
  decltype((DECLVAL(_inner_fold_init&) | ... | static_cast<Sigs*>(nullptr)));

//////////////////////////////////////////////////////////////////////////////////////
// _merge_metadata
//
// After computing a metadata struct for each child sender, all the metadata
// structs must be merged. We use a set of partial specialization of the
// _merge_metadata variable template to merge two metadata structs into one,
// and we use a fold expression to merge all N into one.
template <class Meta1, class Meta2>
extern _undefined<Meta1> _merge_metadata;

// This specialization causes an error to be propagated.
template <class ValsOK, class ErrsOK, class Offsets, class... LeftSigs, class... What>
extern ERROR<What...>& _merge_metadata<_completion_metadata<ValsOK, ErrsOK, Offsets, LeftSigs...>, ERROR<What...>>;

// This overload is selected with the left and right metadata are both for senders
// that have no set_value completion signature.
template <class LeftValsOK,
          class LeftErrsOK,
          class Offsets,
          class... LeftSigs,
          class RightValsOK,
          class RightErrsOK,
          class... RightSigs>
extern _completion_metadata<_mtrue, _mand<LeftErrsOK, RightErrsOK>, _moffsets<>, LeftSigs..., RightSigs...>&
  _merge_metadata<_completion_metadata<LeftValsOK, LeftErrsOK, Offsets, LeftSigs...>,
                  _completion_metadata<RightValsOK, RightErrsOK, _moffsets<>, RightSigs...>>;

// The following two specializations are selected when one of the metadata
// structs is for a sender with no value completions. In that case, the
// when_all can never complete successfully, so drop the other set_value
// completion signature.
template <class LeftValsOK,
          class LeftErrsOK,
          class Offsets,
          class... As,
          class... LeftSigs,
          class RightValsOK,
          class RightErrsOK,
          class... RightSigs>
extern _completion_metadata<
  _mtrue, // There will be no value completion, so values need not be copied.
  _mand<LeftErrsOK, RightErrsOK>,
  _moffsets<>,
  LeftSigs...,
  RightSigs...>& _merge_metadata<_completion_metadata<LeftValsOK, LeftErrsOK, Offsets, set_value_t(As...), LeftSigs...>,
                                 _completion_metadata<RightValsOK, RightErrsOK, _moffsets<>, RightSigs...>>;

template <class LeftValsOK,
          class LeftErrsOK,
          class Offsets,
          class... LeftSigs,
          class RightValsOK,
          class RightErrsOK,
          class... As,
          class... RightSigs>
extern _completion_metadata<_mtrue, // There will be no value completion, so values need not be copied.
                            _mand<LeftErrsOK, RightErrsOK>,
                            _moffsets<>,
                            LeftSigs...,
                            RightSigs...>&
  _merge_metadata<_completion_metadata<LeftValsOK, LeftErrsOK, Offsets, LeftSigs...>,
                  _completion_metadata<RightValsOK, RightErrsOK, _moffsets<>, set_value_t(As...), RightSigs...>>;

template <size_t... Offsets>
_CCCL_INLINE_VAR constexpr size_t _last_offset = (0, ..., Offsets);

template <size_t Count, size_t... Offsets>
using _append_offset = _moffsets<Offsets..., Count + _last_offset<Offsets...>>;

// This overload is selected when both metadata structs are for senders with
// a single value completion. Concatenate the value types.
template <class LeftValsOK,
          class LeftErrsOK,
          auto... Offsets,
          class... As,
          class... LeftSigs,
          class RightValsOK,
          class RightErrsOK,
          class... Bs,
          class... RightSigs>
extern _completion_metadata<_mand<LeftValsOK, RightValsOK>,
                            _mand<LeftErrsOK, RightErrsOK>,
                            _append_offset<sizeof...(Bs), Offsets...>,
                            set_value_t(As..., Bs...), // Concatenate the value types.
                            LeftSigs...,
                            RightSigs...>&
  _merge_metadata<_completion_metadata<LeftValsOK, LeftErrsOK, _moffsets<Offsets...>, set_value_t(As...), LeftSigs...>,
                  _completion_metadata<RightValsOK, RightErrsOK, _moffsets<>, set_value_t(Bs...), RightSigs...>>;

template <class... What, class Other>
extern ERROR<What...>& _merge_metadata<ERROR<What...>, Other>;

// We use a fold expression over the bitwise AND operator to merge all the
// completion metadata structs from the child senders into a single metadata
// struct.
template <class Meta1, class Meta2>
auto operator&(Meta1&, Meta2&) -> decltype(_merge_metadata<Meta1, Meta2>);

// The initial value for the fold.
using _outer_fold_init = _completion_metadata<_mtrue, _mtrue, _moffsets<0ul>, set_value_t(), set_stopped_t()>;

template <class... Sigs>
using _collect_outer = //
  _reduce_completions_t<decltype((DECLVAL(_outer_fold_init&) & ... & DECLVAL(Sigs)))>;

// Extract the first template parameter of the _state_t specialization.
// The first template parameter is the receiver type.
template <class State>
using _rcvr_from_state_t = _mapply<_mpoly_q<_mfront>, State>;

/// The receivers connected to the when_all's sub-operations expose this as
/// their environment. Its `get_stop_token` query returns the token from
/// when_all's stop source. All other queries are forwarded to the outer
/// receiver's environment.
template <class StateZip>
struct _env_t
{
  using _state_t = _unzip<StateZip>;
  using _rcvr_t  = _rcvr_from_state_t<_state_t>;

  _state_t& _state;

  _CCCL_HOST_DEVICE inplace_stop_token query(get_stop_token_t) const noexcept
  {
    return _state._stop_token;
  }

  template <class Tag>
  _CCCL_HOST_DEVICE auto query(Tag) const noexcept -> _query_result_t<Tag, env_of_t<_rcvr_t>>
  {
    return __async::get_env(_state._rcvr).query(Tag());
  }
};

template <class StateZip, size_t Index>
struct _rcvr_t
{
  using receiver_concept = receiver_t;
  using _state_t         = _unzip<StateZip>;

  _state_t& _state;

  template <class... Ts>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_value(Ts&&... ts) noexcept
  {
    constexpr auto idx = _mmake_indices<sizeof...(Ts)>();
    _state.template _set_value<Index>(idx, static_cast<Ts&&>(ts)...);
    _state._arrive();
  }

  template <class Error>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_error(Error&& error) noexcept
  {
    _state._set_error(static_cast<Error&&>(error));
    _state._arrive();
  }

  _CCCL_HOST_DEVICE void set_stopped() noexcept
  {
    _state._set_stopped();
    _state._arrive();
  }

  _CCCL_HOST_DEVICE auto get_env() const noexcept -> _env_t<StateZip>
  {
    return {_state};
  }
};

template <class CvSndr, size_t Idx, class StateZip>
using _inner_completions_ = //
  _mapply_q<_collect_inner, completion_signatures_of_t<CvSndr, _rcvr_t<StateZip, Idx>>>;

template <class CvSndr, size_t Idx, class StateZip>
using _inner_completions = //
  _midentity_or_error_with< //
    _inner_completions_<CvSndr, Idx, StateZip>, //
    WITH_SENDER(CvSndr)>;

enum _estate_t : int
{
  _started,
  _error,
  _stopped
};

/// @brief The data stored in the operation state and refered to
/// by the receiver.
/// @tparam Rcvr The receiver connected to the when_all sender.
/// @tparam CvFn A metafunction to apply cv- and ref-qualifiers to the senders
/// @tparam Sndrs A tuple of the when_all sender's child senders.
template <class Rcvr, class CvFn, class Sndrs>
struct _state_t;

template <class Rcvr, class CvFn, size_t... Idx, class... Sndrs>
struct _state_t<Rcvr, CvFn, _tupl<_mindices<Idx...>, Sndrs...>>
{
  using _completions_offsets_pair_t = //
    _collect_outer< //
      _inner_completions<_minvoke1<CvFn, Sndrs>, Idx, _zip<_state_t>>...>;
  using _completions_t = _mfirst<_completions_offsets_pair_t>;
  using _indices_t     = _mindices<Idx...>;
  using _offsets_t     = _msecond<_completions_offsets_pair_t>;
  using _values_t      = _value_types<_completions_t, _lazy_tuple, _mpoly<_msingle_or<_nil>>::_f>;
  using _errors_t      = _error_types<_completions_t, _variant>;

  using stop_tok_t      = stop_token_of_t<env_of_t<Rcvr>>;
  using stop_callback_t = stop_callback_for_t<stop_tok_t, _on_stop_request>;

  template <size_t Index, size_t... Offsets>
  static constexpr size_t _offset_for(_moffsets<Offsets...>*) noexcept
  {
    constexpr size_t offsets[] = {Offsets..., 0};
    return offsets[Index];
  }

  template <size_t Index, size_t... Jdx, class... Ts>
  _CCCL_HOST_DEVICE void _set_value(_mindices<Jdx...>, Ts&&... ts) noexcept
  {
    constexpr size_t Offset = _offset_for<Index>(static_cast<_offsets_t*>(nullptr));
    if constexpr (!_CUDA_VSTD::is_same_v<_values_t, _nil>)
    {
      if constexpr (_nothrow_decay_copyable<Ts...>)
      {
        (_values.template _emplace<Jdx + Offset>(static_cast<Ts&&>(ts)), ...);
      }
      else
      {
        _CUDAX_TRY( //
          ({ //
            (_values.template _emplace<Jdx + Offset>(static_cast<Ts&&>(ts)), ...);
          }),
          _CUDAX_CATCH(...)( //
            { //
              _set_error(::std::current_exception());
            }))
      }
    }
  }

  template <class Error>
  _CCCL_HOST_DEVICE void _set_error(Error&& _err) noexcept
  {
    // TODO: Use weaker memory orders
    if (_error != _state.exchange(_error))
    {
      _stop_source.request_stop();
      // We won the race, free to write the error into the operation state
      // without worry.
      if constexpr (_nothrow_decay_copyable<Error>)
      {
        _errors.template emplace<_decay_t<Error>>(static_cast<Error&&>(_err));
      }
      else
      {
        _CUDAX_TRY( //
          ({ //
            _errors.template emplace<_decay_t<Error>>(static_cast<Error&&>(_err));
          }),
          _CUDAX_CATCH(...)( //
            { //
              _errors.template emplace<::std::exception_ptr>(::std::current_exception());
            }))
      }
    }
  }

  _CCCL_HOST_DEVICE void _set_stopped() noexcept
  {
    _CUDA_VSTD::underlying_type_t<_estate_t> expected = _started;
    // Transition to the "stopped" state if and only if we're in the
    // "started" state. (If this fails, it's because we're in an
    // error state, which trumps cancellation.)
    if (_state.compare_exchange_strong(expected, static_cast<_CUDA_VSTD::underlying_type_t<_estate_t>>(_stopped)))
    {
      _stop_source.request_stop();
    }
  }

  _CCCL_HOST_DEVICE void _arrive() noexcept
  {
    if (0 == --_count)
    {
      _complete();
    }
  }

  _CCCL_HOST_DEVICE void _complete() noexcept
  {
    // Stop callback is no longer needed. Destroy it.
    _on_stop.destroy();
    // All child operations have completed and arrived at the barrier.
    switch (_state.load(_CUDA_VSTD::memory_order_relaxed))
    {
      case _started:
        if constexpr (!_CUDA_VSTD::is_same_v<_values_t, _nil>)
        {
          // All child operations completed successfully:
          _values.apply(__async::set_value, static_cast<_values_t&&>(_values), static_cast<Rcvr&&>(_rcvr));
        }
        break;
      case _error:
        // One or more child operations completed with an error:
        _errors.visit(__async::set_error, static_cast<_errors_t&&>(_errors), static_cast<Rcvr&&>(_rcvr));
        break;
      case _stopped:
        __async::set_stopped(static_cast<Rcvr&&>(_rcvr));
        break;
      default:;
    }
  }

  Rcvr _rcvr;
  _CUDA_VSTD::atomic<size_t> _count;
  inplace_stop_source _stop_source;
  inplace_stop_token _stop_token{_stop_source.get_token()};
  _CUDA_VSTD::atomic<_CUDA_VSTD::underlying_type_t<_estate_t>> _state{_started};
  _errors_t _errors;
  _values_t _values;
  _lazy<stop_callback_t> _on_stop;
};

/// The operation state for when_all
template <class Rcvr, class CvFn, size_t... Idx, class... Sndrs>
struct _opstate_t<Rcvr, CvFn, _tupl<_mindices<Idx...>, Sndrs...>>
{
  using operation_state_concept = operation_state_t;
  using _sndrs_t                = _minvoke<CvFn, _tuple<Sndrs...>>;
  using _state_t                = _when_all::_state_t<Rcvr, CvFn, _tupl<_mindices<Idx...>, Sndrs...>>;

  using completion_signatures = typename _state_t::_completions_t;
  using _offsets_t            = typename _state_t::_offsets_t;

  // This function object is used to connect all the sub-operations with
  // receivers, each of which knows which elements in the values tuple it
  // is responsible for setting.
  struct _connect_subs_fn
  {
    template <class... CvSndrs>
    _CCCL_HOST_DEVICE auto operator()(_state_t& state, CvSndrs&&... sndrs) const
    {
      using _state_ref_t = _zip<_state_t>;
      if constexpr (_CUDA_VSTD::is_same_v<_offsets_t, _moffsets<>>)
      {
        // When there are no offsets, the when_all sender has no value
        // completions. All child senders can be connected to receivers
        // of the same type.
        return _tupl{__async::connect(static_cast<CvSndrs&&>(sndrs), _rcvr_t<_state_ref_t, 0>{state})...};
      }
      else
      {
        // The offsets are used to determine which elements in the values
        // tuple each receiver is responsible for setting.
        return _tupl{__async::connect(static_cast<CvSndrs&&>(sndrs), _rcvr_t<_state_ref_t, Idx>{state})...};
      }
    }
  };

  // This is a _tuple of operation states for the sub-operations.
  using _sub_opstates_t = _apply_result_t<_connect_subs_fn, _sndrs_t, _state_t&>;

  _state_t _state;
  _sub_opstates_t _sub_ops;

  /// Initialize the data member, connect all the sub-operations and
  /// save the resulting operation states in _sub_ops.
  _CCCL_HOST_DEVICE _opstate_t(_sndrs_t&& sndrs, Rcvr rcvr)
      : _state{static_cast<Rcvr&&>(rcvr), sizeof...(Sndrs)}
      , _sub_ops{sndrs.apply(_connect_subs_fn(), static_cast<_sndrs_t&&>(sndrs), _state)}
  {}

  _CUDAX_IMMOVABLE(_opstate_t);

  /// Start all the sub-operations.
  _CCCL_HOST_DEVICE void start() & noexcept
  {
    // register stop callback:
    _state._on_stop.construct(get_stop_token(__async::get_env(_state._rcvr)), _on_stop_request{_state._stop_source});

    if (_state._stop_source.stop_requested())
    {
      // Manually clean up the stop callback. We won't be starting the
      // sub-operations, so they won't complete and clean up for us.
      _state._on_stop.destroy();

      // Stop has already been requested. Don't bother starting the child
      // operations.
      __async::set_stopped(static_cast<Rcvr&&>(_state._rcvr));
    }
    else
    {
      // Start all the sub-operations.
      _sub_ops.for_each(__async::start, _sub_ops);

      // If there are no sub-operations, we're done.
      if constexpr (sizeof...(Sndrs) == 0)
      {
        _state._complete();
      }
    }
  }
};

template <class... Sndrs>
struct _sndr_t;
} // namespace _when_all

struct when_all_t
{
  template <class... Sndrs>
  _CCCL_HOST_DEVICE _when_all::_sndr_t<Sndrs...> operator()(Sndrs... sndrs) const;
};

// The sender for when_all
template <class... Sndrs>
struct _when_all::_sndr_t
{
  using sender_concept = sender_t;
  using _sndrs_t       = _tuple<Sndrs...>;

  _CCCL_NO_UNIQUE_ADDRESS when_all_t _tag;
  _CCCL_NO_UNIQUE_ADDRESS _ignore _ignore1;
  _sndrs_t _sndrs;

  template <class Rcvr>
  _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) && -> _opstate_t<Rcvr, _cp, _sndrs_t>
  {
    return _opstate_t<Rcvr, _cp, _sndrs_t>(static_cast<_sndrs_t&&>(_sndrs), static_cast<Rcvr&&>(rcvr));
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) const& //
    -> _opstate_t<Rcvr, _cpclr, _sndrs_t>
  {
    return _opstate_t<Rcvr, _cpclr, _sndrs_t>(_sndrs, static_cast<Rcvr&&>(rcvr));
  }
};

template <class... Sndrs>
_CCCL_HOST_DEVICE _when_all::_sndr_t<Sndrs...> when_all_t::operator()(Sndrs... sndrs) const
{
  // If the incoming sender is non-dependent, we can check the completion
  // signatures of the composed sender immediately.
  if constexpr ((_is_non_dependent_sender<Sndrs> && ...))
  {
    using _completions = completion_signatures_of_t<_when_all::_sndr_t<Sndrs...>>;
    static_assert(_is_completion_signatures<_completions>);
  }
  return _when_all::_sndr_t<Sndrs...>{{}, {}, {static_cast<Sndrs&&>(sndrs)...}};
}

_CCCL_GLOBAL_CONSTANT when_all_t when_all{};

} // namespace cuda::experimental::__async

_CCCL_NV_DIAG_DEFAULT(expr_has_no_effect)
_CCCL_DIAG_POP

#include "epilogue.cuh"
