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
#include "queries.cuh"
#include "receiver_with_env.cuh"
#include "tuple.cuh"
#include "utility.cuh"
#include "variant.cuh"

// Must be the last include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
template <class Sch>
struct _sch_env_t
{
  Sch _sch;

  Sch query(get_scheduler_t) const noexcept
  {
    return _sch;
  }
};

_CCCL_GLOBAL_CONSTANT struct start_on_t
{
#ifndef __CUDACC__

private:
#endif

  template <class Rcvr, class Sch, class CvSndr>
  struct _opstate_t
  {
    _CCCL_HOST_DEVICE friend env_of_t<Rcvr> get_env(const _opstate_t* self) noexcept
    {
      return __async::get_env(self->_env_rcvr.rcvr());
    }

    using operation_state_concept = operation_state_t;

    using completion_signatures = //
      transform_completion_signatures<
        completion_signatures_of_t<CvSndr, _receiver_with_env_t<Rcvr, _sch_env_t<Sch>>*>,
        transform_completion_signatures<completion_signatures_of_t<schedule_result_t<Sch>, _opstate_t*>,
                                        __async::completion_signatures<>,
                                        _malways<__async::completion_signatures<>>::_f>>;

    _receiver_with_env_t<Rcvr, _sch_env_t<Sch>> _env_rcvr;
    connect_result_t<schedule_result_t<Sch>, _opstate_t*> _opstate1;
    connect_result_t<CvSndr, _receiver_with_env_t<Rcvr, _sch_env_t<Sch>>*> _opstate2;

    _CCCL_HOST_DEVICE _opstate_t(Sch sch, Rcvr rcvr, CvSndr&& sndr)
        : _env_rcvr{static_cast<Rcvr&&>(rcvr), {sch}}
        , _opstate1{connect(schedule(_env_rcvr._env._sch), this)}
        , _opstate2{connect(static_cast<CvSndr&&>(sndr), &_env_rcvr)}
    {}

    _CUDAX_IMMOVABLE(_opstate_t);

    _CCCL_HOST_DEVICE void start() noexcept
    {
      __async::start(_opstate1);
    }

    _CCCL_HOST_DEVICE void set_value() noexcept
    {
      __async::start(_opstate2);
    }

    template <class Error>
    _CCCL_HOST_DEVICE void set_error(Error&& error) noexcept
    {
      __async::set_error(static_cast<Rcvr&&>(_env_rcvr.rcvr()), static_cast<Error&&>(error));
    }

    _CCCL_HOST_DEVICE void set_stopped() noexcept
    {
      __async::set_stopped(static_cast<Rcvr&&>(_env_rcvr.rcvr()));
    }
  };

  template <class Sch, class Sndr>
  struct _sndr_t;

public:
  template <class Sch, class Sndr>
  _CCCL_HOST_DEVICE auto operator()(Sch sch, Sndr sndr) const noexcept //
    -> _sndr_t<Sch, Sndr>;
} start_on{};

template <class Sch, class Sndr>
struct start_on_t::_sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS start_on_t _tag;
  Sch _sch;
  Sndr _sndr;

  template <class Rcvr>
  _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) && -> _opstate_t<Rcvr, Sch, Sndr>
  {
    return _opstate_t<Rcvr, Sch, Sndr>{_sch, static_cast<Rcvr&&>(rcvr), static_cast<Sndr&&>(_sndr)};
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) const& -> _opstate_t<Rcvr, Sch, const Sndr&>
  {
    return _opstate_t<Rcvr, Sch, const Sndr&>{_sch, static_cast<Rcvr&&>(rcvr), _sndr};
  }

  _CCCL_HOST_DEVICE env_of_t<Sndr> get_env() const noexcept
  {
    return __async::get_env(_sndr);
  }
};

template <class Sch, class Sndr>
_CCCL_HOST_DEVICE auto start_on_t::operator()(Sch sch, Sndr sndr) const noexcept -> start_on_t::_sndr_t<Sch, Sndr>
{
  return _sndr_t<Sch, Sndr>{{}, sch, sndr};
}
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
