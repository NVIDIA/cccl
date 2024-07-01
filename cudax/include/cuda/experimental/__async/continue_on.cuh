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
#include "meta.cuh"
#include "queries.cuh"
#include "tuple.cuh"
#include "utility.cuh"
#include "variant.cuh"

// This must be the last #include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
struct continue_on_t
{
#ifndef __CUDACC__

private:
#endif
  template <class... As>
  using _set_value_tuple_t = _tuple<set_value_t, _decay_t<As>...>;

  template <class Error>
  using _set_error_tuple_t = _tuple<set_error_t, _decay_t<Error>>;

  using _set_stopped_tuple_t = _tuple<set_stopped_t>;

  using _complete_fn = void (*)(void*) noexcept;

  template <class... Ts>
  using _set_value_completion =
    _mif<_nothrow_decay_copyable<Ts...>,
         completion_signatures<set_value_t(_decay_t<Ts>...)>,
         completion_signatures<set_value_t(_decay_t<Ts>...), set_error_t(::std::exception_ptr)>>;

  template <class Error>
  using _set_error_completion =
    _mif<_nothrow_decay_copyable<Error>,
         completion_signatures<set_error_t(_decay_t<Error>)>,
         completion_signatures<set_error_t(_decay_t<Error>), set_error_t(::std::exception_ptr)>>;

  template <class Rcvr, class Result>
  struct _rcvr_t
  {
    using receiver_concept = receiver_t;
    Rcvr _rcvr;
    Result _result;
    _complete_fn _complete;

    template <class Tag, class... As>
    _CCCL_HOST_DEVICE void operator()(Tag, As&... as) noexcept
    {
      Tag()(static_cast<Rcvr&&>(_rcvr), static_cast<As&&>(as)...);
    }

    template <class Tag, class... As>
    _CCCL_HOST_DEVICE void _set_result(Tag, As&&... as) noexcept
    {
      using _tupl_t = _tuple<Tag, _decay_t<As>...>;
      if constexpr (_nothrow_decay_copyable<As...>)
      {
        _result.template emplace<_tupl_t>(Tag(), static_cast<As&&>(as)...);
      }
      else
      {
        _CUDAX_TRY( //
          ({ //
            _result.template emplace<_tupl_t>(Tag(), static_cast<As&&>(as)...);
          }),
          _CUDAX_CATCH(...)( //
            { //
              __async::set_error(static_cast<Rcvr&&>(_rcvr), ::std::current_exception());
            }))
      }
      _complete = +[](void* ptr) noexcept {
        auto& self = *static_cast<_rcvr_t*>(ptr);
        auto& tupl = *static_cast<_tupl_t*>(self._result._ptr());
        tupl.apply(self, tupl);
      };
    }

    _CCCL_HOST_DEVICE void set_value() noexcept
    {
      _complete(this);
    }

    template <class Error>
    _CCCL_HOST_DEVICE void set_error(Error&& error) noexcept
    {
      __async::set_error(static_cast<Rcvr&&>(_rcvr), static_cast<Error&&>(error));
    }

    _CCCL_HOST_DEVICE void set_stopped() noexcept
    {
      __async::set_stopped(static_cast<Rcvr&&>(_rcvr));
    }

    _CCCL_HOST_DEVICE env_of_t<Rcvr> get_env() const noexcept
    {
      return __async::get_env(_rcvr);
    }
  };

  template <class Rcvr, class CvSndr, class Sch>
  struct _opstate_t
  {
    _CCCL_HOST_DEVICE friend auto get_env(const _opstate_t* self) noexcept -> env_of_t<Rcvr>
    {
      return __async::get_env(self->_rcvr._rcvr);
    }

    using operation_state_concept = operation_state_t;
    using _result_t =
      _transform_completion_signatures<completion_signatures_of_t<CvSndr, _opstate_t*>,
                                       _set_value_tuple_t,
                                       _set_error_tuple_t,
                                       _set_stopped_tuple_t,
                                       _variant>;

    // The scheduler contributes error and stopped completions.
    // This causes its set_value_t() completion to be ignored.
    using _scheduler_completions = //
      transform_completion_signatures<completion_signatures_of_t<schedule_result_t<Sch>, _rcvr_t<Rcvr, _result_t>*>,
                                      __async::completion_signatures<>,
                                      _malways<__async::completion_signatures<>>::_f>;

    // The continue_on completions are the scheduler's error
    // and stopped completions, plus the sender's completions
    // with all the result data types decayed.
    using completion_signatures = //
      transform_completion_signatures<completion_signatures_of_t<CvSndr, _opstate_t*>,
                                      _scheduler_completions,
                                      _set_value_completion,
                                      _set_error_completion>;

    _rcvr_t<Rcvr, _result_t> _rcvr;
    connect_result_t<CvSndr, _opstate_t*> _opstate1;
    connect_result_t<schedule_result_t<Sch>, _rcvr_t<Rcvr, _result_t>*> _opstate2;

    _CCCL_HOST_DEVICE _opstate_t(CvSndr&& sndr, Sch sch, Rcvr rcvr)
        : _rcvr{static_cast<Rcvr&&>(rcvr), {}, nullptr}
        , _opstate1{__async::connect(static_cast<CvSndr&&>(sndr), this)}
        , _opstate2{__async::connect(schedule(sch), &_rcvr)}
    {}

    _CUDAX_IMMOVABLE(_opstate_t);

    _CCCL_HOST_DEVICE void start() noexcept
    {
      __async::start(_opstate1);
    }

    template <class... As>
    _CCCL_HOST_DEVICE void set_value(As&&... as) noexcept
    {
      _rcvr._set_result(set_value_t(), static_cast<As&&>(as)...);
      __async::start(_opstate2);
    }

    template <class Error>
    _CCCL_HOST_DEVICE void set_error(Error&& error) noexcept
    {
      _rcvr._set_result(set_error_t(), static_cast<Error&&>(error));
      __async::start(_opstate2);
    }

    _CCCL_HOST_DEVICE void set_stopped() noexcept
    {
      _rcvr._set_result(set_stopped_t());
      __async::start(_opstate2);
    }
  };

  template <class Sndr, class Sch>
  struct _sndr_t;

  template <class Sch>
  struct _closure_t;

public:
  template <class Sndr, class Sch>
  _CCCL_HOST_DEVICE _sndr_t<Sndr, Sch> operator()(Sndr sndr, Sch sch) const noexcept;

  template <class Sch>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE _closure_t<Sch> operator()(Sch sch) const noexcept;
};

template <class Sch>
struct continue_on_t::_closure_t
{
  Sch _sch;

  template <class Sndr>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE friend auto operator|(Sndr sndr, _closure_t&& _self)
  {
    return continue_on_t()(static_cast<Sndr&&>(sndr), static_cast<Sch&&>(_self._sch));
  }
};

template <class Sndr, class Sch>
struct continue_on_t::_sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS continue_on_t _tag;
  Sch _sch;
  Sndr _sndr;

  struct _attrs_t
  {
    _sndr_t* _sndr;

    template <class SetTag>
    _CCCL_HOST_DEVICE auto query(get_completion_scheduler_t<SetTag>) const noexcept
    {
      return _sndr->_sch;
    }

    template <class Query>
    _CCCL_HOST_DEVICE auto query(Query) const //
      -> _query_result_t<Query, env_of_t<Sndr>>
    {
      return __async::get_env(_sndr->_sndr).query(Query{});
    }
  };

  template <class Rcvr>
  _CCCL_HOST_DEVICE _opstate_t<Rcvr, Sndr, Sch> connect(Rcvr rcvr) &&
  {
    return {static_cast<Sndr&&>(_sndr), _sch, static_cast<Rcvr&&>(rcvr)};
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE _opstate_t<Rcvr, const Sndr&, Sch> connect(Rcvr rcvr) const&
  {
    return {_sndr, _sch, static_cast<Rcvr&&>(rcvr)};
  }

  _CCCL_HOST_DEVICE _attrs_t get_env() const noexcept
  {
    return _attrs_t{this};
  }
};

template <class Sndr, class Sch>
_CCCL_HOST_DEVICE auto continue_on_t::operator()(Sndr sndr, Sch sch) const noexcept -> continue_on_t::_sndr_t<Sndr, Sch>
{
  return _sndr_t<Sndr, Sch>{{}, sch, static_cast<Sndr&&>(sndr)};
}

template <class Sch>
_CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE continue_on_t::_closure_t<Sch> continue_on_t::operator()(Sch sch) const noexcept
{
  return _closure_t<Sch>{sch};
}

_CCCL_GLOBAL_CONSTANT continue_on_t continue_on{};
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
