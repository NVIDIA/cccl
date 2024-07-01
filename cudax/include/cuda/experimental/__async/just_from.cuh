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
#include "config.cuh"
#include "cpos.cuh"
#include "rcvr_ref.cuh"
#include "tuple.cuh"
#include "utility.cuh"

// Must be the last include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
// Forward declarations of the just* tag types:
struct just_from_t;
struct just_error_from_t;
struct just_stopped_from_t;

// Map from a disposition to the corresponding tag types:
namespace _detail
{
template <_disposition_t, class Void = void>
extern _undefined<Void> _just_from_tag;
template <class Void>
extern _fn_t<just_from_t>* _just_from_tag<_value, Void>;
template <class Void>
extern _fn_t<just_error_from_t>* _just_from_tag<_error, Void>;
template <class Void>
extern _fn_t<just_stopped_from_t>* _just_from_tag<_stopped, Void>;
} // namespace _detail

struct AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT;
struct A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS;

template <_disposition_t Disposition>
struct _just_from
{
#ifndef __CUDACC__

private:
#endif

  using JustTag = decltype(_detail::_just_from_tag<Disposition>());
  using SetTag  = decltype(_detail::_set_tag<Disposition>());

  using _diag_t = _mif<_CUDA_VSTD::is_same_v<SetTag, set_error_t>,
                       AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT,
                       A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS>;

  template <class... Ts>
  using _error_t = ERROR<WHERE(IN_ALGORITHM, JustTag), WHAT(_diag_t), WITH_COMPLETION_SIGNATURE<SetTag(Ts...)>>;

  struct _probe_fn
  {
    template <class... Ts>
    auto operator()(Ts&&... ts) const noexcept
      -> _mif<_is_valid_signature<SetTag(Ts...)>, completion_signatures<SetTag(Ts...)>, _error_t<Ts...>>;
  };

  template <class Rcvr = receiver_archetype>
  struct _complete_fn
  {
    Rcvr& _rcvr;

    template <class... Ts>
    _CCCL_HOST_DEVICE auto operator()(Ts&&... ts) const noexcept
    {
      SetTag()(static_cast<Rcvr&>(_rcvr), static_cast<Ts&&>(ts)...);
    }
  };

  template <class Rcvr, class Fn>
  struct _opstate
  {
    using operation_state_concept = operation_state_t;
    using completion_signatures   = _call_result_t<Fn, _probe_fn>;
    static_assert(_is_completion_signatures<completion_signatures>);

    Rcvr _rcvr;
    Fn _fn;

    _CCCL_HOST_DEVICE void start() & noexcept
    {
      static_cast<Fn&&>(_fn)(_complete_fn<Rcvr>{_rcvr});
    }
  };

  template <class Fn>
  struct _sndr_t
  {
    using sender_concept = sender_t;

    _CCCL_NO_UNIQUE_ADDRESS JustTag _tag;
    Fn _fn;

    template <class Rcvr>
    _CCCL_HOST_DEVICE _opstate<Rcvr, Fn> connect(Rcvr rcvr) && //
      noexcept(_nothrow_decay_copyable<Rcvr, Fn>)
    {
      return _opstate<Rcvr, Fn>{static_cast<Rcvr&&>(rcvr), static_cast<Fn&&>(_fn)};
    }

    template <class Rcvr>
    _CCCL_HOST_DEVICE _opstate<Rcvr, Fn> connect(Rcvr rcvr) const& //
      noexcept(_nothrow_decay_copyable<Rcvr, Fn const&>)
    {
      return _opstate<Rcvr, Fn>{static_cast<Rcvr&&>(rcvr), _fn};
    }
  };

public:
  template <class Fn>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(Fn fn) const noexcept
  {
    using _completions = _call_result_t<Fn, _probe_fn>;
    static_assert(_is_completion_signatures<_completions>,
                  "The function passed to just_from must return an instance of a specialization of "
                  "completion_signatures<>.");
    return _sndr_t<Fn>{{}, static_cast<Fn&&>(fn)};
  }
};

_CCCL_GLOBAL_CONSTANT struct just_from_t : _just_from<_value>
{
} just_from{};

_CCCL_GLOBAL_CONSTANT struct just_error_from_t : _just_from<_error>
{
} just_error_from{};

_CCCL_GLOBAL_CONSTANT struct just_stopped_from_t : _just_from<_stopped>
{
} just_stopped_from{};
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
