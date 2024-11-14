//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_TYPE_TRAITS
#define __CUDAX_ASYNC_DETAIL_TYPE_TRAITS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{

//////////////////////////////////////////////////////////////////////////////////////////////////
// __decay_t: An efficient implementation for ::std::decay
#if defined(_CCCL_BUILTIN_DECAY)

template <class _Ty>
using __decay_t = _CCCL_BUILTIN_DECAY(_Ty);

#else // ^^^ _CCCL_BUILTIN_DECAY ^^^ / vvv !_CCCL_BUILTIN_DECAY vvv

template <class _Ty>
using __decay_t = _CUDA_VSTD::decay_t<_Ty>;

#endif // _CCCL_BUILTIN_DECAY

//////////////////////////////////////////////////////////////////////////////////////////////////
// __copy_cvref_t: For copying cvref from one type to another
// TODO: This is a temporary implementation. We should merge this file and meta.cuh
// with the facilities in <cuda/std/__type_traits/type_list.h>.
template <class _Ty>
using __cref_t = _Ty const&;

using __cp    = _CUDA_VSTD::__type_self;
using __cpclr = _CUDA_VSTD::__type_quote1<__cref_t>;

template <class _From, class _To>
using __copy_cvref_t = _CUDA_VSTD::__copy_cvref_t<_From, _To>;

template <class _Fn, class... _As>
using __call_result_t = decltype(__declval<_Fn>()(__declval<_As>()...));

template <class _Fn, class... _As>
inline constexpr bool __callable = __type_valid_v<__call_result_t, _Fn, _As...>;

#if defined(__CUDA_ARCH__)
template <class _Fn, class... _As>
inline constexpr bool __nothrow_callable = true;

template <class _Ty, class... _As>
inline constexpr bool __nothrow_constructible = true;

template <class... _As>
inline constexpr bool __nothrow_decay_copyable = true;

template <class... _As>
inline constexpr bool __nothrow_movable = true;

template <class... _As>
inline constexpr bool __nothrow_copyable = true;
#else
template <class _Fn, class... _As>
using __nothrow_callable_ = _CUDA_VSTD::enable_if_t<noexcept(__declval<_Fn>()(__declval<_As>()...))>;

template <class _Fn, class... _As>
inline constexpr bool __nothrow_callable = __type_valid_v<__nothrow_callable_, _Fn, _As...>;

template <class _Ty, class... _As>
using __nothrow_constructible_ = _CUDA_VSTD::enable_if_t<noexcept(_Ty{__declval<_As>()...})>;

template <class _Ty, class... _As>
inline constexpr bool __nothrow_constructible = __type_valid_v<__nothrow_constructible_, _Ty, _As...>;

template <class _Ty>
using __nothrow_decay_copyable_ = _CUDA_VSTD::enable_if_t<noexcept(__decay_t<_Ty>(__declval<_Ty>()))>;

template <class... _As>
inline constexpr bool __nothrow_decay_copyable = (__type_valid_v<__nothrow_decay_copyable_, _As> && ...);

template <class _Ty>
using __nothrow_movable_ = _CUDA_VSTD::enable_if_t<noexcept(_Ty(__declval<_Ty>()))>;

template <class... _As>
inline constexpr bool __nothrow_movable = (__type_valid_v<__nothrow_movable_, _As> && ...);

template <class _Ty>
using __nothrow_copyable_ = _CUDA_VSTD::enable_if_t<noexcept(_Ty(__declval<const _Ty&>()))>;

template <class... _As>
inline constexpr bool __nothrow_copyable = (__type_valid_v<__nothrow_copyable_, _As> && ...);
#endif

template <class... _As>
using __nothrow_decay_copyable_t = _CUDA_VSTD::bool_constant<__nothrow_decay_copyable<_As...>>;
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
