//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_TYPE_TRAITS_CUH
#define __CUDAX_DETAIL_TYPE_TRAITS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <template <class...> class _Fn, class... _Ts>
inline constexpr bool __is_instantiable_with_v = _CUDA_VSTD::_IsValidExpansion<_Fn, _Ts...>::value;

template <class _Ty>
using __decay_copyable_ _CCCL_NODEBUG_ALIAS = decltype(_CUDA_VSTD::decay_t<_Ty>(_CUDA_VSTD::declval<_Ty>()));

template <class... _As>
inline constexpr bool __decay_copyable = (_CUDA_VSTD::is_constructible_v<_CUDA_VSTD::decay_t<_As>, _As> && ...);

#if _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)
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
#else // ^^^ _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC) ^^^ /
      // vvv !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC) vvv
template <class _Fn, class... _As>
inline constexpr bool __nothrow_callable = _CUDA_VSTD::__is_nothrow_callable_v<_Fn, _As...>;

template <class _Ty, class... _As>
inline constexpr bool __nothrow_constructible = _CUDA_VSTD::is_nothrow_constructible_v<_Ty, _As...>;

template <class... _As>
inline constexpr bool __nothrow_decay_copyable =
  (_CUDA_VSTD::is_nothrow_constructible_v<_CUDA_VSTD::decay_t<_As>, _As> && ...);

template <class... _As>
inline constexpr bool __nothrow_movable = (_CUDA_VSTD::is_nothrow_move_constructible_v<_As> && ...);

template <class... _As>
inline constexpr bool __nothrow_copyable = (_CUDA_VSTD::is_nothrow_copy_constructible_v<_As> && ...);
#endif // ^^^ !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC) ^^^

template <class... _As>
using __nothrow_decay_copyable_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::bool_constant<__nothrow_decay_copyable<_As...>>;

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_DETAIL_TYPE_TRAITS_CUH
