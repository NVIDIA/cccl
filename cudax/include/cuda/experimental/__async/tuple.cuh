//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_TUPLE
#define __CUDAX_ASYNC_DETAIL_TUPLE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
template <size_t _Idx, class _Ty>
struct __box
{
  // Too many compiler bugs with [[no_unique_address]] to use it here.
  // E.g., https://github.com/llvm/llvm-project/issues/88077
  // _CCCL_NO_UNIQUE_ADDRESS
  _Ty __value_;
};

template <size_t _Idx, class _Ty>
_CUDAX_TRIVIAL_API constexpr auto __cget(__box<_Idx, _Ty> const& __box) noexcept -> _Ty const&
{
  return __box.__value_;
}

template <class _Idx, class... _Ts>
struct __tupl;

template <size_t... _Idx, class... _Ts>
struct __tupl<_CUDA_VSTD::index_sequence<_Idx...>, _Ts...> : __box<_Idx, _Ts>...
{
  template <class _Fn, class _Self, class... _Us>
  _CUDAX_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) //
    noexcept(__nothrow_callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>...>)
      -> __call_result_t<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>...>
  {
    return static_cast<_Fn&&>(__fn)( //
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__box<_Idx, _Ts>::__value_...);
  }

  template <class _Fn, class _Self, class... _Us>
  _CUDAX_TRIVIAL_API static auto __for_each(_Fn&& __fn, _Self&& __self, _Us&&... __us) //
    noexcept((__nothrow_callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>>
              && ...)) -> _CUDA_VSTD::enable_if_t<(__callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>> && ...)>
  {
    return (
      static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__box<_Idx, _Ts>::__value_),
      ...);
  }
};

template <class... _Ts>
_CUDAX_API __tupl(_Ts...) //
  -> __tupl<_CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>, _Ts...>;

template <class _Fn, class _Tupl, class... _Us>
using __apply_result_t =
  decltype(__declval<_Tupl>().__apply(__declval<_Fn>(), __declval<_Tupl>(), __declval<_Us>()...));

#if _CCCL_COMPILER(MSVC)
template <class... _Ts>
struct __mk_tuple_
{
  using __indices_t = _CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>;
  using type        = __tupl<__indices_t, _Ts...>;
};

template <class... _Ts>
using __tuple = typename __mk_tuple_<_Ts...>::type;
#else
template <class... _Ts>
using __tuple = __tupl<_CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>, _Ts...>;
#endif

template <class... _Ts>
using __decayed_tuple = __tuple<__decay_t<_Ts>...>;
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
