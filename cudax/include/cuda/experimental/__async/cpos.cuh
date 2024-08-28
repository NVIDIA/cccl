//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_CPOS_H
#define __CUDAX_ASYNC_DETAIL_CPOS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/config.cuh>
#include <cuda/experimental/__async/env.cuh>
#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/utility.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
struct receiver_t
{};

struct operation_state_t
{};

struct sender_t
{};

struct scheduler_t
{};

template <class Ty>
using _sender_concept_t = typename _remove_reference_t<Ty>::sender_concept;

template <class Ty>
using _receiver_concept_t = typename _remove_reference_t<Ty>::receiver_concept;

template <class Ty>
using _scheduler_concept_t = typename _remove_reference_t<Ty>::scheduler_concept;

template <class Ty>
_CCCL_INLINE_VAR constexpr bool _is_sender = _mvalid_q<_sender_concept_t, Ty>;

template <class Ty>
_CCCL_INLINE_VAR constexpr bool _is_receiver = _mvalid_q<_receiver_concept_t, Ty>;

template <class Ty>
_CCCL_INLINE_VAR constexpr bool _is_scheduler = _mvalid_q<_scheduler_concept_t, Ty>;

_CCCL_GLOBAL_CONSTANT struct set_value_t
{
  template <class Rcvr, class... Ts>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(Rcvr&& rcvr, Ts&&... ts) const noexcept
    -> decltype(static_cast<Rcvr&&>(rcvr).set_value(static_cast<Ts&&>(ts)...))
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<Rcvr&&>(rcvr).set_value(static_cast<Ts&&>(ts)...)), void>);
    static_assert(noexcept(static_cast<Rcvr&&>(rcvr).set_value(static_cast<Ts&&>(ts)...)));
    static_cast<Rcvr&&>(rcvr).set_value(static_cast<Ts&&>(ts)...);
  }

  template <class Rcvr, class... Ts>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(Rcvr* rcvr, Ts&&... ts) const noexcept
    -> decltype(static_cast<Rcvr&&>(*rcvr).set_value(static_cast<Ts&&>(ts)...))
  {
    static_assert(
      _CUDA_VSTD::is_same_v<decltype(static_cast<Rcvr&&>(*rcvr).set_value(static_cast<Ts&&>(ts)...)), void>);
    static_assert(noexcept(static_cast<Rcvr&&>(*rcvr).set_value(static_cast<Ts&&>(ts)...)));
    static_cast<Rcvr&&>(*rcvr).set_value(static_cast<Ts&&>(ts)...);
  }
} set_value{};

_CCCL_GLOBAL_CONSTANT struct set_error_t
{
  template <class Rcvr, class E>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto
  operator()(Rcvr&& rcvr, E&& e) const noexcept -> decltype(static_cast<Rcvr&&>(rcvr).set_error(static_cast<E&&>(e)))
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<Rcvr&&>(rcvr).set_error(static_cast<E&&>(e))), void>);
    static_assert(noexcept(static_cast<Rcvr&&>(rcvr).set_error(static_cast<E&&>(e))));
    static_cast<Rcvr&&>(rcvr).set_error(static_cast<E&&>(e));
  }

  template <class Rcvr, class E>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto
  operator()(Rcvr* rcvr, E&& e) const noexcept -> decltype(static_cast<Rcvr&&>(*rcvr).set_error(static_cast<E&&>(e)))
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<Rcvr&&>(*rcvr).set_error(static_cast<E&&>(e))), void>);
    static_assert(noexcept(static_cast<Rcvr&&>(*rcvr).set_error(static_cast<E&&>(e))));
    static_cast<Rcvr&&>(*rcvr).set_error(static_cast<E&&>(e));
  }
} set_error{};

_CCCL_GLOBAL_CONSTANT struct set_stopped_t
{
  template <class Rcvr>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto
  operator()(Rcvr&& rcvr) const noexcept -> decltype(static_cast<Rcvr&&>(rcvr).set_stopped())
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<Rcvr&&>(rcvr).set_stopped()), void>);
    static_assert(noexcept(static_cast<Rcvr&&>(rcvr).set_stopped()));
    static_cast<Rcvr&&>(rcvr).set_stopped();
  }

  template <class Rcvr>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto
  operator()(Rcvr* rcvr) const noexcept -> decltype(static_cast<Rcvr&&>(*rcvr).set_stopped())
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<Rcvr&&>(*rcvr).set_stopped()), void>);
    static_assert(noexcept(static_cast<Rcvr&&>(*rcvr).set_stopped()));
    static_cast<Rcvr&&>(*rcvr).set_stopped();
  }
} set_stopped{};

_CCCL_GLOBAL_CONSTANT struct start_t
{
  template <class OpState>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(OpState& opstate) const noexcept -> decltype(opstate.start())
  {
    static_assert(!_is_error<typename OpState::completion_signatures>);
    static_assert(_CUDA_VSTD::is_same_v<decltype(opstate.start()), void>);
    static_assert(noexcept(opstate.start()));
    opstate.start();
  }
} start{};

_CCCL_GLOBAL_CONSTANT struct connect_t
{
  template <class Sndr, class Rcvr>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(Sndr&& sndr, Rcvr&& rcvr) const
    noexcept(noexcept(static_cast<Sndr&&>(sndr).connect(static_cast<Rcvr&&>(rcvr))))
      -> decltype(static_cast<Sndr&&>(sndr).connect(static_cast<Rcvr&&>(rcvr)))
  {
    // using opstate_t     = decltype(static_cast<Sndr&&>(sndr).connect(static_cast<Rcvr&&>(rcvr)));
    // using completions_t = typename opstate_t::completion_signatures;
    // static_assert(_is_completion_signatures<completions_t>);

    return static_cast<Sndr&&>(sndr).connect(static_cast<Rcvr&&>(rcvr));
  }
} connect{};

_CCCL_GLOBAL_CONSTANT struct schedule_t
{
  template <class Sch>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto
  operator()(Sch&& sch) const noexcept -> decltype(static_cast<Sch&&>(sch).schedule())
  {
    static_assert(noexcept(static_cast<Sch&&>(sch).schedule()));
    return static_cast<Sch&&>(sch).schedule();
  }
} schedule{};

struct receiver_archetype
{
  using receiver_concept = receiver_t;

  template <class... Ts>
  void set_value(Ts&&...) noexcept;

  template <class Error>
  void set_error(Error&&) noexcept;

  void set_stopped() noexcept;

  env<> get_env() const noexcept;
};

template <class Sndr, class Rcvr>
using connect_result_t = decltype(connect(DECLVAL(Sndr), DECLVAL(Rcvr)));

template <class Sndr, class Rcvr = receiver_archetype>
using completion_signatures_of_t = typename connect_result_t<Sndr, Rcvr>::completion_signatures;

template <class Sch>
using schedule_result_t = decltype(schedule(DECLVAL(Sch)));

template <class Sndr, class Rcvr>
_CCCL_INLINE_VAR constexpr bool _nothrow_connectable = noexcept(connect(DECLVAL(Sndr), DECLVAL(Rcvr)));

// handy enumerations for keeping type names readable
enum _disposition_t
{
  _value,
  _error,
  _stopped
};

namespace _detail
{
template <_disposition_t, class Void = void>
extern _undefined<Void> _set_tag;
template <class Void>
extern _fn_t<set_value_t>* _set_tag<_value, Void>;
template <class Void>
extern _fn_t<set_error_t>* _set_tag<_error, Void>;
template <class Void>
extern _fn_t<set_stopped_t>* _set_tag<_stopped, Void>;
} // namespace _detail
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
