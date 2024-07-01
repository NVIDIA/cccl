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

#include "completion_signatures.cuh"
#include "cpos.cuh"
#include "exception.cuh"
#include "rcvr_ref.cuh"
#include "tuple.cuh"
#include "variant.cuh"

// This must be the last #include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
// Declare types to use for diagnostics:
struct FUNCTION_MUST_RETURN_A_SENDER;

// Forward-declate the let_* algorithm tag types:
struct let_value_t;
struct let_error_t;
struct let_stopped_t;

// Map from a disposition to the corresponding tag types:
namespace _detail
{
template <_disposition_t, class Void = void>
extern _undefined<Void> _let_tag;
template <class Void>
extern _fn_t<let_value_t>* _let_tag<_value, Void>;
template <class Void>
extern _fn_t<let_error_t>* _let_tag<_error, Void>;
template <class Void>
extern _fn_t<let_stopped_t>* _let_tag<_stopped, Void>;
} // namespace _detail

template <_disposition_t Disposition>
struct _let
{
#ifndef __CUDACC__

private:
#endif
  using LetTag = decltype(_detail::_let_tag<Disposition>());
  using SetTag = decltype(_detail::_set_tag<Disposition>());

  template <class...>
  using _empty_tuple = _tuple<>;

  /// @brief Computes the type of a variant of tuples to hold the results of
  /// the predecessor sender.
  template <class CvSndr, class Rcvr>
  using _results =
    _gather_completion_signatures<completion_signatures_of_t<CvSndr, Rcvr>, SetTag, _decayed_tuple, _empty_tuple, _variant>;

  template <class Fn, class Rcvr>
  struct _opstate_fn
  {
    template <class... As>
    using _f = connect_result_t<_call_result_t<Fn, _decay_t<As>&...>, _rcvr_ref_t<Rcvr&>>;
  };

  /// @brief Computes the type of a variant of operation states to hold
  /// the second operation state.
  template <class CvSndr, class Fn, class Rcvr>
  using _opstate2_t =
    _gather_completion_signatures<completion_signatures_of_t<CvSndr, Rcvr>,
                                  SetTag,
                                  _opstate_fn<Fn, Rcvr>::template _f,
                                  _empty_tuple,
                                  _variant>;

  template <class Fn, class Rcvr>
  struct _completions_fn
  {
    using _error_non_sender_return = //
      ERROR<WHERE(IN_ALGORITHM, LetTag), WHAT(FUNCTION_MUST_RETURN_A_SENDER), WITH_FUNCTION(Fn)>;

    template <class Ty>
    using _ensure_sender = //
      _mif<_is_sender<Ty> || _is_error<Ty>, Ty, _error_non_sender_return>;

    template <class... As>
    using _error_not_callable_with = //
      ERROR<WHERE(IN_ALGORITHM, LetTag), WHAT(FUNCTION_IS_NOT_CALLABLE), WITH_FUNCTION(Fn), WITH_ARGUMENTS(As...)>;

    // This computes the result of calling the function with the
    // predecessor sender's results. If the function is not callable with
    // the results, it returns an ERROR.
    template <class... As>
    using _call_result = _minvoke<_mtry_quote<_call_result_t, _error_not_callable_with<As...>>, Fn, _decay_t<As>&...>;

    // This computes the completion signatures of sender returned by the
    // function when called with the given arguments. It return an ERROR if
    // the function is not callable with the arguments or if the function
    // returns a non-sender.
    template <class... As>
    using _f = _mtry_invoke_q<completion_signatures_of_t, _ensure_sender<_call_result<As...>>, _rcvr_ref_t<Rcvr&>>;
  };

  /// @brief Computes the completion signatures of the
  /// `let_(value|error|stopped)` sender.
  template <class CvSndr, class Fn, class Rcvr>
  using _completions =
    _gather_completion_signatures<completion_signatures_of_t<CvSndr, Rcvr>,
                                  SetTag,
                                  _completions_fn<Fn, Rcvr>::template _f,
                                  _default_completions,
                                  _mbind_front<_mtry_quote<_concat_completion_signatures>, _eptr_completion>::_f>;

  /// @brief The `let_(value|error|stopped)` operation state.
  /// @tparam CvSndr The cvref-qualified predecessor sender type.
  /// @tparam Fn The function to be called when the predecessor sender
  /// completes.
  /// @tparam Rcvr The receiver connected to the `let_(value|error|stopped)`
  /// sender.
  template <class Rcvr, class CvSndr, class Fn>
  struct _opstate_t
  {
    _CCCL_HOST_DEVICE friend env_of_t<Rcvr> get_env(const _opstate_t* self) noexcept
    {
      return __async::get_env(self->_rcvr);
    }

    using operation_state_concept = operation_state_t;
    using completion_signatures   = _completions<CvSndr, Fn, Rcvr>;

    // Don't try to compute the type of the variant of operation states
    // if the computation of the completion signatures failed.
    using _deferred_opstate_fn = _mbind_back<_mtry_quote<_opstate2_t>, CvSndr, Fn, Rcvr>;
    using _opstate_variant_fn  = _mif<_is_error<completion_signatures>, _malways<_empty>, _deferred_opstate_fn>;
    using _opstate_variant_t   = _mtry_invoke<_opstate_variant_fn>;

    Rcvr _rcvr;
    Fn _fn;
    _results<CvSndr, _opstate_t*> _result;
    connect_result_t<CvSndr, _opstate_t*> _opstate1;
    _opstate_variant_t _opstate2;

    _CCCL_HOST_DEVICE _opstate_t(CvSndr&& sndr, Fn fn, Rcvr rcvr) noexcept(
      _nothrow_decay_copyable<Fn, Rcvr> && _nothrow_connectable<CvSndr, _opstate_t*>)
        : _rcvr(static_cast<Rcvr&&>(rcvr))
        , _fn(static_cast<Fn&&>(fn))
        , _opstate1(__async::connect(static_cast<CvSndr&&>(sndr), this))
    {}

    _CCCL_HOST_DEVICE void start() noexcept
    {
      __async::start(_opstate1);
    }

    template <class Tag, class... As>
    _CCCL_HOST_DEVICE void _complete(Tag, As&&... as) noexcept
    {
      if constexpr (_CUDA_VSTD::is_same_v<Tag, SetTag>)
      {
        _CUDAX_TRY( //
          ({ //
            // Store the results so the lvalue refs we pass to the function
            // will be valid for the duration of the async op.
            auto& tupl = _result.template emplace<_decayed_tuple<As...>>(static_cast<As&&>(as)...);
            if constexpr (!_is_error<completion_signatures>)
            {
              // Call the function with the results and connect the resulting
              // sender, storing the operation state in _opstate2.
              auto& nextop = _opstate2.emplace_from(
                __async::connect, tupl.apply(static_cast<Fn&&>(_fn), tupl), __async::_rcvr_ref(_rcvr));
              __async::start(nextop);
            }
          }),
          _CUDAX_CATCH(...)( //
            { //
              __async::set_error(static_cast<Rcvr&&>(_rcvr), ::std::current_exception());
            }))
      }
      else
      {
        // Forward the completion to the receiver unchanged.
        Tag()(static_cast<Rcvr&&>(_rcvr), static_cast<As&&>(as)...);
      }
    }

    template <class... As>
    _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_value(As&&... as) noexcept
    {
      _complete(set_value_t(), static_cast<As&&>(as)...);
    }

    template <class Error>
    _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_error(Error&& error) noexcept
    {
      _complete(set_error_t(), static_cast<Error&&>(error));
    }

    _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_stopped() noexcept
    {
      _complete(set_stopped_t());
    }
  };

  /// @brief The `let_(value|error|stopped)` sender.
  /// @tparam Sndr The predecessor sender.
  /// @tparam Fn The function to be called when the predecessor sender
  /// completes.
  template <class Sndr, class Fn>
  struct _sndr_t
  {
    using sender_concept = sender_t;
    _CCCL_NO_UNIQUE_ADDRESS LetTag _tag;
    Fn _fn;
    Sndr _sndr;

    template <class Rcvr>
    _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) && noexcept(
      _nothrow_constructible<_opstate_t<Rcvr, Sndr, Fn>, Sndr, Fn, Rcvr>) -> _opstate_t<Rcvr, Sndr, Fn>
    {
      return _opstate_t<Rcvr, Sndr, Fn>(static_cast<Sndr&&>(_sndr), static_cast<Fn&&>(_fn), static_cast<Rcvr&&>(rcvr));
    }

    template <class Rcvr>
    _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) const& noexcept( //
      _nothrow_constructible<_opstate_t<Rcvr, const Sndr&, Fn>,
                             const Sndr&,
                             const Fn&,
                             Rcvr>) //
      -> _opstate_t<Rcvr, const Sndr&, Fn>
    {
      return _opstate_t<Rcvr, const Sndr&, Fn>(_sndr, _fn, static_cast<Rcvr&&>(rcvr));
    }

    _CCCL_HOST_DEVICE env_of_t<Sndr> get_env() const noexcept
    {
      return __async::get_env(_sndr);
    }
  };

  template <class Fn>
  struct _closure_t
  {
    using LetTag = decltype(_detail::_let_tag<Disposition>());
    Fn _fn;

    template <class Sndr>
    _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto operator()(Sndr sndr) const //
      -> _call_result_t<LetTag, Sndr, Fn>
    {
      return LetTag()(static_cast<Sndr&&>(sndr), _fn);
    }

    template <class Sndr>
    _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE friend auto operator|(Sndr sndr, const _closure_t& _self) //
      -> _call_result_t<LetTag, Sndr, Fn>
    {
      return LetTag()(static_cast<Sndr&&>(sndr), _self._fn);
    }
  };

public:
  template <class Sndr, class Fn>
  _CCCL_HOST_DEVICE _sndr_t<Sndr, Fn> operator()(Sndr sndr, Fn fn) const
  {
    // If the incoming sender is non-dependent, we can check the completion
    // signatures of the composed sender immediately.
    if constexpr (_is_non_dependent_sender<Sndr>)
    {
      using _completions = completion_signatures_of_t<_sndr_t<Sndr, Fn>>;
      static_assert(_is_completion_signatures<_completions>);
    }
    return _sndr_t<Sndr, Fn>{{}, static_cast<Fn&&>(fn), static_cast<Sndr&&>(sndr)};
  }

  template <class Fn>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(Fn fn) const noexcept
  {
    return _closure_t<Fn>{static_cast<Fn&&>(fn)};
  }
};

_CCCL_GLOBAL_CONSTANT struct let_value_t : _let<_value>
{
} let_value{};

_CCCL_GLOBAL_CONSTANT struct let_error_t : _let<_error>
{
} let_error{};

_CCCL_GLOBAL_CONSTANT struct let_stopped_t : _let<_stopped>
{
} let_stopped{};
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
