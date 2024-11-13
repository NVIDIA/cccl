//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_JUST
#define __CUDAX_ASYNC_DETAIL_JUST

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/tuple.cuh>
#include <cuda/experimental/__async/utility.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
// Forward declarations of the just* tag types:
struct just_t;
struct just_error_t;
struct just_stopped_t;

// Map from a disposition to the corresponding tag types:
namespace __detail
{
template <__disposition_t, class _Void = void>
extern __undefined<_Void> __just_tag;
template <class _Void>
extern __fn_t<just_t>* __just_tag<__value, _Void>;
template <class _Void>
extern __fn_t<just_error_t>* __just_tag<__error, _Void>;
template <class _Void>
extern __fn_t<just_stopped_t>* __just_tag<__stopped, _Void>;
} // namespace __detail

template <__disposition_t _Disposition>
struct __just
{
#if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#endif // _CCCL_CUDA_COMPILER_NVCC

  using _JustTag = decltype(__detail::__just_tag<_Disposition>());
  using _SetTag  = decltype(__detail::__set_tag<_Disposition>());

  template <class _Rcvr, class... _Ts>
  struct __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using completion_signatures   = __async::completion_signatures<_SetTag(_Ts...)>;
    _Rcvr __rcvr_;
    __tuple<_Ts...> __values_;

    struct __complete_fn
    {
      __opstate_t* __self_;

      _CUDAX_API void operator()(_Ts&... __ts) const noexcept
      {
        _SetTag()(static_cast<_Rcvr&&>(__self_->__rcvr_), static_cast<_Ts&&>(__ts)...);
      }
    };

    _CUDAX_API void start() & noexcept
    {
      __values_.__apply(__complete_fn{this}, __values_);
    }
  };

  template <class... _Ts>
  struct __sndr_t
  {
    using sender_concept        = sender_t;
    using completion_signatures = __async::completion_signatures<_SetTag(_Ts...)>;

    _CCCL_NO_UNIQUE_ADDRESS _JustTag __tag_;
    __tuple<_Ts...> __values_;

    template <class _Rcvr>
    _CUDAX_API __opstate_t<_Rcvr, _Ts...> connect(_Rcvr __rcvr) && //
      noexcept(__nothrow_decay_copyable<_Rcvr, _Ts...>)
    {
      return __opstate_t<_Rcvr, _Ts...>{static_cast<_Rcvr&&>(__rcvr), static_cast<__tuple<_Ts...>&&>(__values_)};
    }

    template <class _Rcvr>
    _CUDAX_API __opstate_t<_Rcvr, _Ts...> connect(_Rcvr __rcvr) const& //
      noexcept(__nothrow_decay_copyable<_Rcvr, _Ts const&...>)
    {
      return __opstate_t<_Rcvr, _Ts...>{static_cast<_Rcvr&&>(__rcvr), __values_};
    }
  };

public:
  template <class... _Ts>
  _CUDAX_TRIVIAL_API auto operator()(_Ts... __ts) const noexcept
  {
    return __sndr_t<_Ts...>{_JustTag{}, {{static_cast<_Ts&&>(__ts)}...}};
  }
};

_CCCL_GLOBAL_CONSTANT struct just_t : __just<__value>
{
} just{};

_CCCL_GLOBAL_CONSTANT struct just_error_t : __just<__error>
{
} just_error{};

_CCCL_GLOBAL_CONSTANT struct just_stopped_t : __just<__stopped>
{
} just_stopped{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
