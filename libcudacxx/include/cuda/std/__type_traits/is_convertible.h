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

#include <cuda/std/__type_traits/enable_if.h>
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
// gcc 13's builin doesn't properly implement some function conversions
#elif _CCCL_CHECK_BUILTIN(is_convertible) && !_CCCL_COMPILER(GCC, <, 14) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX(14)
#  define _CCCL_BUILTIN_IS_CONVERTIBLE_TO(...) __is_convertible(__VA_ARGS__)
#endif // ^^^ has builtin is_convertible_to

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_CONVERTIBLE_TO) && !defined(_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK)

template <class _Fm, class _To>
inline constexpr bool is_convertible_v = _CCCL_BUILTIN_IS_CONVERTIBLE_TO(_Fm, _To);

#  if _CCCL_COMPILER(MSVC) // Workaround for DevCom-1627396
template <class _Tp>
inline constexpr bool is_convertible_v<_Tp&, volatile _Tp&> = true;

template <class _Tp>
inline constexpr bool is_convertible_v<volatile _Tp&, volatile _Tp&> = true;

template <class _Tp>
inline constexpr bool is_convertible_v<_Tp&, const volatile _Tp&> = true;

template <class _Tp>
inline constexpr bool is_convertible_v<volatile _Tp&, const volatile _Tp&> = true;
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

// We can't use template variable here, because gcc emits errors regarding us touching private members.
template <class _From, class _To>
_CCCL_API false_type __is_convertible_test(long);
template <class _From,
          class _To,
          class = decltype(::cuda::std::__is_convertible_imp::__test_convert<_To>(::cuda::std::declval<_From>()))>
_CCCL_API true_type __is_convertible_test(int);

template <class _Tp, bool _IsArray = is_array_v<_Tp>, bool _IsFunction = is_function_v<_Tp>, bool _IsVoid = is_void_v<_Tp>>
inline constexpr int __is_array_function_or_void_v = 0;
template <class _Tp>
inline constexpr int __is_array_function_or_void_v<_Tp, true, false, false> = 1;
template <class _Tp>
inline constexpr int __is_array_function_or_void_v<_Tp, false, true, false> = 2;
template <class _Tp>
inline constexpr int __is_array_function_or_void_v<_Tp, false, false, true> = 3;
} // namespace __is_convertible_imp

template <class _T1,
          class _T2,
          int _T1_is_array_function_or_void = __is_convertible_imp::__is_array_function_or_void_v<_T1>,
          int _T2_is_array_function_or_void = __is_convertible_imp::__is_array_function_or_void_v<_T2>>
inline constexpr bool __is_convertible_fallback_v =
  decltype(::cuda::std::__is_convertible_imp::__is_convertible_test<_T1, _T2>(0))::value;

template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 0, 1> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 1, 1> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 2, 1> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 3, 1> = false;

template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 0, 2> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 1, 2> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 2, 2> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 3, 2> = false;

template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 0, 3> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 1, 3> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 2, 3> = false;
template <class _T1, class _T2>
inline constexpr bool __is_convertible_fallback_v<_T1, _T2, 3, 3> = true;

template <class _Fm, class _To>
inline constexpr bool is_convertible_v = __is_convertible_fallback_v<_Fm, _To>;

#endif // ^^^ !_CCCL_BUILTIN_IS_CONVERTIBLE_TO ^^^

template <class _Fm, class _To>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_convertible : bool_constant<is_convertible_v<_Fm, _To>>
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_CONVERTIBLE_H
