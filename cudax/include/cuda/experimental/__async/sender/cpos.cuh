//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__async/sender/fwd.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
// make the completion tags equality comparable
template <__disposition_t _Disposition>
struct __completion_tag
{
  template <__disposition_t _OtherDisposition>
  _CUDAX_TRIVIAL_API constexpr auto operator==(__completion_tag<_OtherDisposition>) const noexcept -> bool
  {
    return _Disposition == _OtherDisposition;
  }

  template <__disposition_t _OtherDisposition>
  _CUDAX_TRIVIAL_API constexpr auto operator!=(__completion_tag<_OtherDisposition>) const noexcept -> bool
  {
    return _Disposition != _OtherDisposition;
  }

  static constexpr __disposition_t __disposition = _Disposition;
};

struct set_value_t : __completion_tag<__value>
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
};

struct set_error_t : __completion_tag<__error>
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
};

struct set_stopped_t : __completion_tag<__stopped>
{
  template <class _Rcvr>
  _CUDAX_TRIVIAL_API auto operator()(_Rcvr&& __rcvr) const noexcept
    -> decltype(static_cast<_Rcvr&&>(__rcvr).set_stopped())
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(static_cast<_Rcvr&&>(__rcvr).set_stopped()), void>);
    static_assert(noexcept(static_cast<_Rcvr&&>(__rcvr).set_stopped()));
    static_cast<_Rcvr&&>(__rcvr).set_stopped();
  }
};

struct start_t
{
  template <class _OpState>
  _CUDAX_TRIVIAL_API auto operator()(_OpState& __opstate) const noexcept -> decltype(__opstate.start())
  {
    static_assert(_CUDA_VSTD::is_same_v<decltype(__opstate.start()), void>);
    static_assert(noexcept(__opstate.start()));
    __opstate.start();
  }
};

// connect
struct connect_t
{
  template <class _Sndr, class _Rcvr>
  _CUDAX_TRIVIAL_API auto operator()(_Sndr&& __sndr, _Rcvr&& __rcvr) const
    noexcept(noexcept(static_cast<_Sndr&&>(__sndr).connect(static_cast<_Rcvr&&>(__rcvr))))
      -> decltype(static_cast<_Sndr&&>(__sndr).connect(static_cast<_Rcvr&&>(__rcvr)))
  {
    return static_cast<_Sndr&&>(__sndr).connect(static_cast<_Rcvr&&>(__rcvr));
  }
};

struct schedule_t
{
  template <class _Sch>
  _CUDAX_TRIVIAL_API auto operator()(_Sch&& __sch) const noexcept -> decltype(static_cast<_Sch&&>(__sch).schedule())
  {
    static_assert(noexcept(static_cast<_Sch&&>(__sch).schedule()));
    return static_cast<_Sch&&>(__sch).schedule();
  }
};

_CCCL_GLOBAL_CONSTANT set_value_t set_value{};
_CCCL_GLOBAL_CONSTANT set_error_t set_error{};
_CCCL_GLOBAL_CONSTANT set_stopped_t set_stopped{};
_CCCL_GLOBAL_CONSTANT start_t start{};
_CCCL_GLOBAL_CONSTANT connect_t connect{};
_CCCL_GLOBAL_CONSTANT schedule_t schedule{};

template <class _Sndr, class _Rcvr>
using connect_result_t _CCCL_NODEBUG_ALIAS = decltype(connect(declval<_Sndr>(), declval<_Rcvr>()));

template <class _Sch>
using schedule_result_t _CCCL_NODEBUG_ALIAS = decltype(schedule(declval<_Sch>()));

template <class _Sndr, class _Rcvr>
inline constexpr bool __nothrow_connectable = noexcept(connect(declval<_Sndr>(), declval<_Rcvr>()));
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
