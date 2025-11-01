//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_CONVERTIBLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_CONVERTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(is_convertible_to) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_CONVERTIBLE_TO(...) __is_convertible_to(__VA_ARGS__)
#elif _CCCL_CHECK_BUILTIN(is_convertible)
#  define _CCCL_BUILTIN_IS_CONVERTIBLE_TO(...) __is_convertible(__VA_ARGS__)
#endif // ^^^ has builtin is_convertible_to

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_CONVERTIBLE_TO) && !defined(_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK)

template <class _Tp, class _Up>
inline constexpr bool is_convertible_v = _CCCL_BUILTIN_IS_CONVERTIBLE_TO(_Tp, _Up);

#  if _CCCL_COMPILER(MSVC) // Workaround for DevCom-1627396
template <class _Ty>
inline constexpr bool is_convertible_v<_Ty&, volatile _Ty&> = true;

template <class _Ty>
inline constexpr bool is_convertible_v<volatile _Ty&, volatile _Ty&> = true;

template <class _Ty>
inline constexpr bool is_convertible_v<_Ty&, const volatile _Ty&> = true;

template <class _Ty>
inline constexpr bool is_convertible_v<volatile _Ty&, const volatile _Ty&> = true;
#  endif // _CCCL_COMPILER(MSVC)

#else // ^^^ _CCCL_BUILTIN_IS_CONVERTIBLE_TO ^^^ / vvv !_CCCL_BUILTIN_IS_CONVERTIBLE_TO vvv

namespace __is_convertible_imp
{

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(volatile_func_param_deprecated)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(volatile_func_param_deprecated)
_CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated-volatile")

template <class _Tp>
_CCCL_API inline void __test_convert(_Tp);

_CCCL_END_NV_DIAG_SUPPRESS()
_CCCL_DIAG_POP

template <class _From, class _To, class = void>
inline constexpr bool __is_convertible_test_v = false;

template <class _From, class _To>
inline constexpr bool __is_convertible_test_v<
  _From,
  _To,
  decltype(::cuda::std::__is_convertible_imp::__test_convert<_To>(::cuda::std::declval<_From>()))> = true;

template <class _Tp, bool _IsArray = is_array_v<_Tp>, bool _IsFunction = is_function_v<_Tp>, bool _IsVoid = is_void_v<_Tp>>
inline constexpr int __is_array_function_or_void_v = 0;
template <class _Tp>
inline constexpr int __is_array_function_or_void_v<_Tp, true, false, false> = 1;
template <class _Tp>
inline constexpr int __is_array_function_or_void_v<_Tp, false, true, false> = 2;
template <class _Tp>
inline constexpr int __is_array_function_or_void_v<_Tp, false, false, true> = 3;
} // namespace __is_convertible_imp

template <class _Tp,
          class _Up,
          unsigned _Tp_is_array_function_or_void = __is_convertible_imp::__is_array_function_or_void_v<_Tp>,
          unsigned _Up_is_array_function_or_void = __is_convertible_imp::__is_array_function_or_void_v<_Up>>
inline constexpr bool __is_convertible_fallback_v = __is_convertible_imp::__is_convertible_test_v<_Tp, _Up>;

template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 0, 1> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 1, 1> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 2, 1> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 3, 1> = false;

template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 0, 2> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 1, 2> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 2, 2> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 3, 2> = false;

template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 0, 3> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 1, 3> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 2, 3> = false;
template <class _Tp, class _Up>
inline constexpr bool __is_convertible_fallback_v<_Tp, _Up, 3, 3> = true;

template <class _From, class _To>
inline constexpr bool is_convertible_v = __is_convertible_fallback_v<_From, _To>;

#endif // ^^^ !_CCCL_BUILTIN_IS_CONVERTIBLE_TO ^^^

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_convertible : bool_constant<is_convertible_v<_Tp, _Up>>
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_CONVERTIBLE_H
