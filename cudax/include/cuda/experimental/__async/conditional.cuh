/*
 * Copyright (c) 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "completion_signatures.cuh"
#include "config.cuh"
#include "just_from.cuh"
#include "meta.cuh"
#include "type_traits.cuh"
#include "variant.cuh"

// This must be the last #include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
struct _cond_t
{
  template <class Pred, class Then, class Else>
  struct _data
  {
    Pred pred_;
    Then then_;
    Else else_;
  };

  template <class... Args>
  _CCCL_HOST_DEVICE static auto _mk_complete_fn(Args&&... args) noexcept
  {
    return [&](auto sink) noexcept {
      return sink(static_cast<Args&&>(args)...);
    };
  }

  template <class... Args>
  using _just_from_t = decltype(just_from(_cond_t::_mk_complete_fn(DECLVAL(Args)...)));

  template <class Sndr, class Rcvr, class Pred, class Then, class Else>
  struct _opstate
  {
    using operation_state_concept = operation_state_t;

    _CCCL_HOST_DEVICE friend env_of_t<Rcvr> get_env(const _opstate* self) noexcept
    {
      return get_env(self->_rcvr);
    }

    template <class... Args>
    using _value_t = //
      transform_completion_signatures<
        completion_signatures_of_t<_call_result_t<Then, _just_from_t<Args...>>, _rcvr_ref_t<Rcvr&>>,
        completion_signatures_of_t<_call_result_t<Else, _just_from_t<Args...>>, _rcvr_ref_t<Rcvr&>>>;

    template <class... Args>
    using _opstate_t = //
      _mlist< //
        connect_result_t<_call_result_t<Then, _just_from_t<Args...>>, _rcvr_ref_t<Rcvr&>>,
        connect_result_t<_call_result_t<Else, _just_from_t<Args...>>, _rcvr_ref_t<Rcvr&>>>;

    using completion_signatures = //
      transform_completion_signatures_of<Sndr, _opstate*, __async::completion_signatures<>, _value_t>;

    _CCCL_HOST_DEVICE _opstate(Sndr&& sndr, Rcvr&& rcvr, _data<Pred, Then, Else>&& data)
        : _rcvr{static_cast<Rcvr&&>(rcvr)}
        , _data{static_cast<_cond_t::_data<Pred, Then, Else>>(data)}
        , _op{__async::connect(static_cast<Sndr&&>(sndr), this)}
    {}

    _CCCL_HOST_DEVICE void start() noexcept
    {
      __async::start(_op);
    }

    template <class... Args>
    _CCCL_HOST_DEVICE void set_value(Args&&... args) noexcept
    {
      if (static_cast<Pred&&>(_data.pred_)(args...))
      {
        auto& op = _ops.emplace_from(
          connect,
          static_cast<Then&&>(_data.then_)(just_from(_cond_t::_mk_complete_fn(static_cast<Args&&>(args)...))),
          _rcvr_ref(_rcvr));
        __async::start(op);
      }
      else
      {
        auto& op = _ops.emplace_from(
          connect,
          static_cast<Else&&>(_data.else_)(just_from(_cond_t::_mk_complete_fn(static_cast<Args&&>(args)...))),
          _rcvr_ref(_rcvr));
        __async::start(op);
      }
    }

    template <class Error>
    _CCCL_HOST_DEVICE void set_error(Error&& err) noexcept
    {
      __async::set_error(static_cast<Rcvr&&>(_rcvr), static_cast<Error&&>(err));
    }

    _CCCL_HOST_DEVICE void set_stopped() noexcept
    {
      __async::set_stopped(static_cast<Rcvr&&>(_rcvr));
    }

    Rcvr _rcvr;
    _cond_t::_data<Pred, Then, Else> _data;
    connect_result_t<Sndr, _opstate*> _op;
    _value_types<completion_signatures_of_t<Sndr, _opstate*>, _opstate_t, _mconcat_into_q<_variant>::_f> _ops;
  };

  template <class Sndr, class Pred, class Then, class Else>
  struct _sndr;

  template <class Pred, class Then, class Else>
  struct _closure
  {
    _cond_t::_data<Pred, Then, Else> _data;

    template <class Sndr>
    _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto _mk_sender(Sndr&& sndr) //
      -> _sndr<Sndr, Pred, Then, Else>;

    template <class Sndr>
    _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto operator()(Sndr sndr) //
      -> _sndr<Sndr, Pred, Then, Else>
    {
      return _mk_sender(_CUDA_VSTD::move(sndr));
    }

    template <class Sndr>
    _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE friend auto operator|(Sndr sndr, _closure&& _self) //
      -> _sndr<Sndr, Pred, Then, Else>
    {
      return _self._mk_sender(_CUDA_VSTD::move(sndr));
    }
  };

  template <class Sndr, class Pred, class Then, class Else>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto operator()(Sndr sndr, Pred pred, Then then, Else _else) const //
    -> _sndr<Sndr, Pred, Then, Else>;

  template <class Pred, class Then, class Else>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto operator()(Pred pred, Then then, Else _else) const
  {
    return _closure<Pred, Then, Else>{
      {static_cast<Pred&&>(pred), static_cast<Then&&>(then), static_cast<Else&&>(_else)}};
  }
};

template <class Sndr, class Pred, class Then, class Else>
struct _cond_t::_sndr
{
  _cond_t _tag;
  _cond_t::_data<Pred, Then, Else> _data;
  Sndr _sndr;

  template <class Rcvr>
  _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) && -> _opstate<Sndr, Rcvr, Pred, Then, Else>
  {
    return {
      static_cast<Sndr&&>(_sndr), static_cast<Rcvr&&>(rcvr), static_cast<_cond_t::_data<Pred, Then, Else>&&>(_data)};
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) const& -> _opstate<Sndr const&, Rcvr, Pred, Then, Else>
  {
    return {_sndr, static_cast<Rcvr&&>(rcvr), static_cast<_cond_t::_data<Pred, Then, Else>&&>(_data)};
  }

  _CCCL_HOST_DEVICE env_of_t<Sndr> get_env() const noexcept
  {
    return __async::get_env(_sndr);
  }
};

template <class Sndr, class Pred, class Then, class Else>
_CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto _cond_t::operator()(Sndr sndr, Pred pred, Then then, Else _else) const //
  -> _sndr<Sndr, Pred, Then, Else>
{
  if constexpr (_is_non_dependent_sender<Sndr>)
  {
    using _completions = completion_signatures_of_t<_sndr<Sndr, Pred, Then, Else>>;
    static_assert(_is_completion_signatures<_completions>);
  }

  return _sndr<Sndr, Pred, Then, Else>{
    {}, {static_cast<Pred&&>(pred), static_cast<Then&&>(then), static_cast<Else&&>(_else)}, static_cast<Sndr&&>(sndr)};
}

template <class Pred, class Then, class Else>
template <class Sndr>
_CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto _cond_t::_closure<Pred, Then, Else>::_mk_sender(Sndr&& sndr) //
  -> _sndr<Sndr, Pred, Then, Else>
{
  if constexpr (_is_non_dependent_sender<Sndr>)
  {
    using _completions = completion_signatures_of_t<_sndr<Sndr, Pred, Then, Else>>;
    static_assert(_is_completion_signatures<_completions>);
  }

  return _sndr<Sndr, Pred, Then, Else>{
    {}, static_cast<_cond_t::_data<Pred, Then, Else>&&>(_data), static_cast<Sndr&&>(sndr)};
}

using conditional_t = _cond_t;
_CCCL_GLOBAL_CONSTANT conditional_t conditional{};
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
