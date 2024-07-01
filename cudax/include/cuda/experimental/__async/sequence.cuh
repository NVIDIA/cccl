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
#include "lazy.cuh"
#include "rcvr_ref.cuh"
#include "variant.cuh"

// This must be the last #include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
struct _seq
{
  template <class Rcvr, class Sndr1, class Sndr2>
  struct _args
  {
    using _rcvr_t  = Rcvr;
    using _sndr1_t = Sndr1;
    using _sndr2_t = Sndr2;
  };

  template <class Zip>
  struct _opstate
  {
    using operation_state_concept = operation_state_t;

    using _args_t  = _unzip<Zip>; // _unzip<Zip> is _args<Rcvr, Sndr1, Sndr2>
    using _rcvr_t  = typename _args_t::_rcvr_t;
    using _sndr1_t = typename _args_t::_sndr1_t;
    using _sndr2_t = typename _args_t::_sndr2_t;

    using completion_signatures = //
      transform_completion_signatures_of< //
        _sndr1_t,
        _opstate*,
        completion_signatures_of_t<_sndr2_t, _rcvr_ref_t<_rcvr_t&>>,
        _malways<__async::completion_signatures<>>::_f>; // swallow the first sender's value completions

    _CCCL_HOST_DEVICE friend env_of_t<_rcvr_t> get_env(const _opstate* self) noexcept
    {
      return __async::get_env(self->_rcvr);
    }

    _rcvr_t _rcvr;
    connect_result_t<_sndr1_t, _opstate*> _op1;
    connect_result_t<_sndr2_t, _rcvr_ref_t<_rcvr_t&>> _op2;

    _CCCL_HOST_DEVICE _opstate(_sndr1_t&& sndr1, _sndr2_t&& sndr2, _rcvr_t&& rcvr)
        : _rcvr(static_cast<_rcvr_t&&>(rcvr))
        , _op1(__async::connect(static_cast<_sndr1_t&&>(sndr1), this))
        , _op2(__async::connect(static_cast<_sndr2_t&&>(sndr2), _rcvr_ref(_rcvr)))
    {}

    _CCCL_HOST_DEVICE void start() noexcept
    {
      __async::start(_op1);
    }

    template <class... Values>
    _CCCL_HOST_DEVICE void set_value(Values&&...) && noexcept
    {
      __async::start(_op2);
    }

    template <class Error>
    _CCCL_HOST_DEVICE void set_error(Error&& err) && noexcept
    {
      __async::set_error(static_cast<_rcvr_t&&>(_rcvr), static_cast<Error&&>(err));
    }

    _CCCL_HOST_DEVICE void set_stopped() && noexcept
    {
      __async::set_stopped(static_cast<_rcvr_t&&>(_rcvr));
    }
  };

  template <class Sndr1, class Sndr2>
  struct _sndr;

  template <class Sndr1, class Sndr2>
  _CCCL_HOST_DEVICE auto operator()(Sndr1 sndr1, Sndr2 sndr2) const -> _sndr<Sndr1, Sndr2>;
};

template <class Sndr1, class Sndr2>
struct _seq::_sndr
{
  using sender_concept = sender_t;
  using _sndr1_t       = Sndr1;
  using _sndr2_t       = Sndr2;

  template <class Rcvr>
  _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) &&
  {
    using _opstate_t = _opstate<_zip<_args<Rcvr, Sndr1, Sndr2>>>;
    return _opstate_t{static_cast<Sndr1&&>(_sndr1), static_cast<Sndr2>(_sndr2), static_cast<Rcvr&&>(rcvr)};
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) const&
  {
    using _opstate_t = _opstate<_zip<_args<Rcvr, const Sndr1&, const Sndr2&>>>;
    return _opstate_t{_sndr1, _sndr2, static_cast<Rcvr&&>(rcvr)};
  }

  _CCCL_HOST_DEVICE env_of_t<Sndr2> get_env() const noexcept
  {
    return __async::get_env(_sndr2);
  }

  _CCCL_NO_UNIQUE_ADDRESS _seq _tag;
  _CCCL_NO_UNIQUE_ADDRESS _ignore _ign;
  _sndr1_t _sndr1;
  _sndr2_t _sndr2;
};

template <class Sndr1, class Sndr2>
_CCCL_HOST_DEVICE auto _seq::operator()(Sndr1 sndr1, Sndr2 sndr2) const -> _sndr<Sndr1, Sndr2>
{
  return _sndr<Sndr1, Sndr2>{{}, {}, static_cast<Sndr1&&>(sndr1), static_cast<Sndr2&&>(sndr2)};
}

using sequence_t = _seq;
_CCCL_GLOBAL_CONSTANT sequence_t sequence{};
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
