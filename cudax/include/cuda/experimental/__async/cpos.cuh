//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_CPOS
#define __CUDAX_ASYNC_DETAIL_CPOS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/env.cuh>
#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/utility.cuh>
#include <cuda/experimental/__detail/config.cuh>

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

template <class _Ty>
using __sender_concept_t = typename _CUDA_VSTD::remove_reference_t<_Ty>::sender_concept;

template <class _Ty>
using __receiver_concept_t = typename _CUDA_VSTD::remove_reference_t<_Ty>::receiver_concept;

template <class _Ty>
using __scheduler_concept_t = typename _CUDA_VSTD::remove_reference_t<_Ty>::scheduler_concept;

template <class _Ty>
inline constexpr bool __is_sender = __type_valid_v<__sender_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_receiver = __type_valid_v<__receiver_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_scheduler = __type_valid_v<__scheduler_concept_t, _Ty>;

_CCCL_GLOBAL_CONSTANT struct set_value_t
{
  template <class _Rcvr, class... _Ts>
  _CUDAX_TRIVIAL_API auto operator()(_Rcvr&& __rcvr, _Ts&&... __ts) const noexcept
    -> decltype(static_cast<_Rcvr&&>(__rcvr).set_value(static_cast<_Ts&&>(__ts)...))
  {
    static_assert(
      _CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(__rcvr).set_value(static_cast<_Ts&&>(__ts)...)), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(__rcvr).set_value(static_cast<_Ts&&>(__ts)...)));
    static_cast<_Rcvr&&>(__rcvr).set_value(static_cast<_Ts&&>(__ts)...);
  }

  template <class _Rcvr, class... _Ts>
  _CUDAX_TRIVIAL_API auto operator()(_Rcvr* __rcvr, _Ts&&... __ts) const noexcept
    -> decltype(static_cast<_Rcvr&&>(*__rcvr).set_value(static_cast<_Ts&&>(__ts)...))
  {
    static_assert(
      _CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(*__rcvr).set_value(static_cast<_Ts&&>(__ts)...)), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(*__rcvr).set_value(static_cast<_Ts&&>(__ts)...)));
    static_cast<_Rcvr&&>(*__rcvr).set_value(static_cast<_Ts&&>(__ts)...);
  }
} set_value{};

_CCCL_GLOBAL_CONSTANT struct set_error_t
{
  template <class _Rcvr, class _Ey>
  _CUDAX_TRIVIAL_API auto operator()(_Rcvr&& __rcvr, _Ey&& __e) const noexcept
    -> decltype(static_cast<_Rcvr&&>(__rcvr).set_error(static_cast<_Ey&&>(__e)))
  {
    static_assert(
      _CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(__rcvr).set_error(static_cast<_Ey&&>(__e))), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(__rcvr).set_error(static_cast<_Ey&&>(__e))));
    static_cast<_Rcvr&&>(__rcvr).set_error(static_cast<_Ey&&>(__e));
  }

  template <class _Rcvr, class _Ey>
  _CUDAX_TRIVIAL_API auto operator()(_Rcvr* __rcvr, _Ey&& __e) const noexcept
    -> decltype(static_cast<_Rcvr&&>(*__rcvr).set_error(static_cast<_Ey&&>(__e)))
  {
    static_assert(
      _CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(*__rcvr).set_error(static_cast<_Ey&&>(__e))), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(*__rcvr).set_error(static_cast<_Ey&&>(__e))));
    static_cast<_Rcvr&&>(*__rcvr).set_error(static_cast<_Ey&&>(__e));
  }
} set_error{};

_CCCL_GLOBAL_CONSTANT struct set_stopped_t
{
  template <class _Rcvr>
  _CUDAX_TRIVIAL_API auto
  operator()(_Rcvr&& __rcvr) const noexcept -> decltype(static_cast<_Rcvr&&>(__rcvr).set_stopped())
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(__rcvr).set_stopped()), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(__rcvr).set_stopped()));
    static_cast<_Rcvr&&>(__rcvr).set_stopped();
  }

  template <class _Rcvr>
  _CUDAX_TRIVIAL_API auto
  operator()(_Rcvr* __rcvr) const noexcept -> decltype(static_cast<_Rcvr&&>(*__rcvr).set_stopped())
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(*__rcvr).set_stopped()), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(*__rcvr).set_stopped()));
    static_cast<_Rcvr&&>(*__rcvr).set_stopped();
  }
} set_stopped{};

_CCCL_GLOBAL_CONSTANT struct start_t
{
  template <class _OpState>
  _CUDAX_TRIVIAL_API auto operator()(_OpState& __opstate) const noexcept -> decltype(__opstate.start())
  {
    static_assert(!__type_is_error<typename _OpState::completion_signatures>);
    static_assert(_CUDA_VSTD::is_same_v<decltype(__opstate.start()), void>);
    static_assert(noexcept(__opstate.start()));
    __opstate.start();
  }
} start{};

_CCCL_GLOBAL_CONSTANT struct connect_t
{
  template <class _Sndr, class _Rcvr>
  _CUDAX_TRIVIAL_API auto operator()(_Sndr&& __sndr, _Rcvr&& __rcvr) const
    noexcept(noexcept(static_cast<_Sndr&&>(__sndr).connect(static_cast<_Rcvr&&>(__rcvr))))
      -> decltype(static_cast<_Sndr&&>(__sndr).connect(static_cast<_Rcvr&&>(__rcvr)))
  {
    // using __opstate_t     = decltype(static_cast<_Sndr&&>(__sndr).connect(static_cast<_Rcvr&&>(__rcvr)));
    // using completions_t = typename __opstate_t::completion_signatures;
    // static_assert(__is_completion_signatures<completions_t>);

    return static_cast<_Sndr&&>(__sndr).connect(static_cast<_Rcvr&&>(__rcvr));
  }
} connect{};

_CCCL_GLOBAL_CONSTANT struct schedule_t
{
  template <class _Sch>
  _CUDAX_TRIVIAL_API auto operator()(_Sch&& __sch) const noexcept -> decltype(static_cast<_Sch&&>(__sch).schedule())
  {
    static_assert(noexcept(static_cast<_Sch&&>(__sch).schedule()));
    return static_cast<_Sch&&>(__sch).schedule();
  }
} schedule{};

struct receiver_archetype
{
  using receiver_concept = receiver_t;

  template <class... _Ts>
  void set_value(_Ts&&...) noexcept;

  template <class _Error>
  void set_error(_Error&&) noexcept;

  void set_stopped() noexcept;

  env<> get_env() const noexcept;
};

template <class _Sndr, class _Rcvr>
using connect_result_t = decltype(connect(__declval<_Sndr>(), __declval<_Rcvr>()));

template <class _Sndr, class _Rcvr = receiver_archetype>
using completion_signatures_of_t = typename connect_result_t<_Sndr, _Rcvr>::completion_signatures;

template <class _Sch>
using schedule_result_t = decltype(schedule(__declval<_Sch>()));

template <class _Sndr, class _Rcvr>
inline constexpr bool __nothrow_connectable = noexcept(connect(__declval<_Sndr>(), __declval<_Rcvr>()));

// handy enumerations for keeping type names readable
enum __disposition_t
{
  __value,
  __error,
  __stopped
};

namespace __detail
{
template <__disposition_t, class _Void = void>
extern __undefined<_Void> __set_tag;
template <class _Void>
extern __fn_t<set_value_t>* __set_tag<__value, _Void>;
template <class _Void>
extern __fn_t<set_error_t>* __set_tag<__error, _Void>;
template <class _Void>
extern __fn_t<set_stopped_t>* __set_tag<__stopped, _Void>;
} // namespace __detail
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
