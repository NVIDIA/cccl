//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_BASIC_SENDER_H
#define __CUDAX_ASYNC_DETAIL_BASIC_SENDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/config.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/utility.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
template <class Data, class Rcvr>
struct state
{
  Data data;
  Rcvr receiver;
};

struct receiver_defaults
{
  using receiver_concept = __async::receiver_t;

  template <class Rcvr, class... Args>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE static auto set_value(_ignore, Rcvr& rcvr, Args&&... args) noexcept
    -> __async::completion_signatures<__async::set_value_t(Args...)>
  {
    __async::set_value(static_cast<Rcvr&&>(rcvr), static_cast<Args&&>(args)...);
    return {};
  }

  template <class Rcvr, class Error>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE static auto
  set_error(_ignore, Rcvr& rcvr, Error&& err) noexcept -> __async::completion_signatures<__async::set_error_t(Error)>
  {
    __async::set_error(static_cast<Rcvr&&>(rcvr), static_cast<Error&&>(err));
    return {};
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE static auto
  set_stopped(_ignore, Rcvr& rcvr) noexcept -> __async::completion_signatures<__async::set_stopped_t()>
  {
    __async::set_stopped(static_cast<Rcvr&&>(rcvr));
    return {};
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE static decltype(auto) get_env(_ignore, const Rcvr& rcvr) noexcept
  {
    return __async::get_env(rcvr);
  }
};

template <class Data, class Rcvr>
struct basic_receiver
{
  using receiver_concept = __async::receiver_t;
  using _rcvr_t          = typename Data::receiver_tag;
  state<Data, Rcvr>& state_;

  template <class... Args>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_value(Args&&... args) noexcept
  {
    _rcvr_t::set_value(state_.data, state_.receiver, (Args&&) args...);
  }

  template <class Error>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_error(Error&& err) noexcept
  {
    _rcvr_t::set_error(state_.data, state_.receiver, (Error&&) err);
  }

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_stopped() noexcept
  {
    _rcvr_t::set_stopped(state_.data, state_.receiver);
  }

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE decltype(auto) get_env() const noexcept
  {
    return _rcvr_t::get_env(state_.data, state_.receiver);
  }
};

template <class Rcvr>
_CCCL_INLINE_VAR constexpr bool has_no_environment = _CUDA_VSTD::is_same_v<Rcvr, receiver_archetype>;

template <bool HasStopped, class Data, class Rcvr>
struct _mk_completions
{
  using _rcvr_t = typename Data::receiver_tag;

  template <class... Args>
  using _set_value_t =
    decltype(+*_rcvr_t::set_value(_declval<Data&>(), _declval<receiver_archetype&>(), _declval<Args>()...));

  template <class Error>
  using _set_error_t =
    decltype(+*_rcvr_t::set_error(_declval<Data&>(), _declval<receiver_archetype&>(), _declval<Error>()));

  using _set_stopped_t = __async::completion_signatures<>;
};

template <class Data, class Rcvr>
struct _mk_completions<true, Data, Rcvr> : _mk_completions<false, Data, Rcvr>
{
  using _rcvr_t = typename Data::receiver_tag;

  using _set_stopped_t = decltype(+*_rcvr_t::set_stopped(_declval<Data&>(), _declval<receiver_archetype&>()));
};

template <class...>
using _ignore_value_signature = __async::completion_signatures<>;

template <class>
using _ignore_error_signature = __async::completion_signatures<>;

template <class Completions>
constexpr bool _has_stopped =
  !_CUDA_VSTD::is_same_v<__async::completion_signatures<>,
                         __async::transform_completion_signatures<Completions,
                                                                  __async::completion_signatures<>,
                                                                  _ignore_value_signature,
                                                                  _ignore_error_signature>>;

template <bool PotentiallyThrowing, class Rcvr>
void set_current_exception_if([[maybe_unused]] Rcvr& rcvr) noexcept
{
  if constexpr (PotentiallyThrowing)
  {
    __async::set_error(static_cast<Rcvr&&>(rcvr), ::std::current_exception());
  }
}

// A generic type that holds the data for an async operation, and
// that provides a `start` method for enqueuing the work.
template <class Sndr, class Data, class Rcvr>
struct basic_opstate
{
  using _rcvr_t        = basic_receiver<Data, Rcvr>;
  using _completions_t = completion_signatures_of_t<Sndr, _rcvr_t>;
  using _traits_t      = _mk_completions<_has_stopped<_completions_t>, Data, Rcvr>;

  using completion_signatures =
    transform_completion_signatures<_completions_t,
                                    __async::completion_signatures<>, // TODO
                                    _traits_t::template _set_value_t,
                                    _traits_t::template _set_error_t,
                                    typename _traits_t::_set_stopped_t>;

  _CCCL_HOST_DEVICE basic_opstate(Sndr&& sndr, Data data, Rcvr rcvr)
      : state_{static_cast<Data&&>(data), static_cast<Rcvr&&>(rcvr)}
      , op_(__async::connect(static_cast<Sndr&&>(sndr), _rcvr_t{state_}))
  {}

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void start() noexcept
  {
    __async::start(op_);
  }

  state<Data, Rcvr> state_;
  __async::connect_result_t<Sndr, _rcvr_t> op_;
};

template <class Sndr, class Rcvr>
_CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto _make_opstate(Sndr sndr, Rcvr rcvr)
{
  auto [tag, data, child] = static_cast<Sndr&&>(sndr);
  using data_t            = decltype(data);
  using child_t           = decltype(child);
  return basic_opstate(static_cast<child_t&&>(child), static_cast<data_t&&>(data), static_cast<Rcvr&&>(rcvr));
}

template <class Data, class... Sndrs>
_CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto
_get_attrs(int, const Data& data, const Sndrs&... sndrs) noexcept -> decltype(data.get_attrs(sndrs...))
{
  return data.get_attrs(sndrs...);
}

template <class Data, class... Sndrs>
_CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto
_get_attrs(long, const Data& data, const Sndrs&... sndrs) noexcept -> decltype(__async::get_env(sndrs...))
{
  return __async::get_env(sndrs...);
}

template <class Data, class... Sndrs>
struct basic_sender;

template <class Data, class Sndr>
struct basic_sender<Data, Sndr>
{
  using sender_concept = __async::sender_t;
  using _tag_t         = typename Data::sender_tag;
  using _rcvr_t        = typename Data::receiver_tag;

  _CCCL_NO_UNIQUE_ADDRESS _tag_t _tag_;
  Data data_;
  Sndr sndr_;

  // Connect the sender to the receiver (the continuation) and
  // return the state_type object for this operation.
  template <class Rcvr>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto connect(Rcvr rcvr) &&
  {
    return _make_opstate(static_cast<basic_sender&&>(*this), static_cast<Rcvr&&>(rcvr));
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE auto connect(Rcvr rcvr) const&
  {
    return _make_opstate(*this, static_cast<Rcvr&&>(rcvr));
  }

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE decltype(auto) get_env() const noexcept
  {
    return __async::_get_attrs(0, data_, sndr_);
  }
};

template <class Data, class... Sndrs>
basic_sender(_ignore, Data, Sndrs...) -> basic_sender<Data, Sndrs...>;

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
