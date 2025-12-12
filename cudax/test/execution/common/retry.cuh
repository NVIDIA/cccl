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

#include <cuda/std/concepts>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/execution.cuh>

namespace _retry_detail
{
namespace ex = ::cuda::experimental::execution;

template <class From, class To>
using _copy_cvref_t = ::cuda::std::__copy_cvref_t<From, To>;

// _conv needed so we can emplace construct non-movable types into
// a cuda::std::optional.
template <class F>
struct _conv
{
  using result_type = decltype(::cuda::std::declval<F>()());

  operator result_type() &&
  {
    return static_cast<F&&>(f_)();
  }

  F f_;
};

template <class F>
_conv(F) -> _conv<F>;

///////////////////////////////////////////////////////////////////////////////
// retry algorithm:
template <class S, class R>
struct _opstate;

// pass through all customizations except set_error, which retries the operation.
template <class S, class R>
struct _retry_receiver
{
  using receiver_concept = ex::receiver_t;

  template <class... Ts>
  void set_value(Ts&&... ts) && noexcept
  {
    ex::set_value(::cuda::std::move(o_->r_), static_cast<Ts&&>(ts)...);
  }

  template <class Error>
  void set_error(Error&&) && noexcept
  {
    o_->_retry(); // This causes the op to be retried
  }

  void set_stopped() && noexcept
  {
    ex::set_stopped(static_cast<R&&>(o_->r_));
  }

  [[nodiscard]]
  auto get_env() const noexcept -> ex::env_of_t<R>
  {
    return ex::get_env(o_->r_);
  }

  _opstate<S, R>* o_;
};

// Hold the nested operation state in an optional so we can
// re-construct and re-start it if the operation fails.
template <class S, class R>
struct _opstate
{
  using operation_state_concept = ex::operation_state_t;
  using _nested_op_t            = ex::connect_result_t<S&, _retry_receiver<S, R>>;

  explicit _opstate(S s, R r)
      : s_(static_cast<S&&>(s))
      , r_(static_cast<R&&>(r))
      , o_{_connect()}
  {}

  _opstate(_opstate&&) = delete;

  [[nodiscard]] auto _connect() noexcept
  {
    return _conv{[this] {
      return ex::connect(s_, _retry_receiver<S, R>{this});
    }};
  }

  void _retry() noexcept
  {
    _CCCL_TRY
    {
      o_.emplace(_connect()); // potentially throwing
      ex::start(*o_);
    }
    _CCCL_CATCH_ALL
    {
      ex::set_error(static_cast<R&&>(r_), ex::current_exception());
    }
  }

  void start() & noexcept
  {
    ex::start(*o_);
  }

private:
  friend struct _retry_receiver<S, R>;
  S s_;
  R r_;
  ::cuda::std::optional<_nested_op_t> o_;
};

struct _swallow_signature
{
  template <class... _Ts>
  _CCCL_CONSTEVAL auto operator()() const noexcept
  {
    return ex::completion_signatures{};
  }
};

template <class S>
struct _retry_sender
{
  using sender_concept = ex::sender_t;

  explicit _retry_sender(S s)
      : s_(static_cast<S&&>(s))
  {}

  template <class Self, class... Env>
  static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    return ex::transform_completion_signatures(
      ex::get_child_completion_signatures<Self&, S, Env...>(),
      {},
      _swallow_signature{},
      {},
      ex::completion_signatures<ex::set_error_t(ex::exception_ptr)>{});
  }

  template <class R>
  [[nodiscard]] auto connect(R r) && -> _opstate<S, R>
  {
    return _opstate<S, R>{::cuda::std::move(*this).s_, ::cuda::std::move(r)};
  }

  template <class R>
  [[nodiscard]] auto connect(R r) const& -> _opstate<S, R>
  {
    return _opstate<S, R>{s_, ::cuda::std::move(r)};
  }

  auto get_env() const noexcept -> ex::env_of_t<S>
  {
    return ex::get_env(s_);
  }

private:
  S s_;
};
} // namespace _retry_detail

struct retry_t
{
  template <class S>
  [[nodiscard]] auto operator()(S s) const -> _retry_detail::_retry_sender<S>
  {
    return _retry_detail::_retry_sender<S>{static_cast<S&&>(s)};
  }
};

inline constexpr retry_t retry{};
