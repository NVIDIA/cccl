//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CONVERTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CONVERTIBLE_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_CONVERTIBLE_TO) && !defined(_LIBCUDACXX_USE_IS_CONVERTIBLE_FALLBACK)

template <class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_convertible
    : public integral_constant<bool, _CCCL_BUILTIN_IS_CONVERTIBLE_TO(_T1, _T2)>
{};

#  if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _T1, class _T2>
_LIBCUDACXX_INLINE_VAR constexpr bool is_convertible_v = _CCCL_BUILTIN_IS_CONVERTIBLE_TO(_T1, _T2);
#  endif

#  ifdef _CCCL_COMPILER_MSVC // Workaround for DevCom-1627396
template <class _Ty>
struct is_convertible<_Ty&, volatile _Ty&> : true_type
{};

template <class _Ty>
struct is_convertible<volatile _Ty&, volatile _Ty&> : true_type
{};

template <class _Ty>
struct is_convertible<_Ty&, const volatile _Ty&> : true_type
{};

template <class _Ty>
struct is_convertible<volatile _Ty&, const volatile _Ty&> : true_type
{};

template <class _Ty>
_LIBCUDACXX_INLINE_VAR constexpr bool is_convertible_v<_Ty&, volatile _Ty&> = true;

template <class _Ty>
_LIBCUDACXX_INLINE_VAR constexpr bool is_convertible_v<volatile _Ty&, volatile _Ty&> = true;

template <class _Ty>
_LIBCUDACXX_INLINE_VAR constexpr bool is_convertible_v<_Ty&, const volatile _Ty&> = true;

template <class _Ty>
_LIBCUDACXX_INLINE_VAR constexpr bool is_convertible_v<volatile _Ty&, const volatile _Ty&> = true;
#  endif // _CCCL_COMPILER_MSVC

#else // ^^^ _CCCL_BUILTIN_IS_CONVERTIBLE_TO ^^^ / vvv !_CCCL_BUILTIN_IS_CONVERTIBLE_TO vvv

namespace __is_convertible_imp
{

_CCCL_NV_DIAG_SUPPRESS(3013) // a volatile function parameter is deprecated
template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI void __test_convert(_Tp);
_CCCL_NV_DIAG_DEFAULT(3013) // a volatile function parameter is deprecated

template <class _From, class _To, class = void>
struct __is_convertible_test : public false_type
{};

template <class _From, class _To>
struct __is_convertible_test<
  _From,
  _To,
  decltype(_CUDA_VSTD::__is_convertible_imp::__test_convert<_To>(_CUDA_VSTD::declval<_From>()))> : public true_type
{};

template <class _Tp,
          bool _IsArray    = is_array<_Tp>::value,
          bool _IsFunction = is_function<_Tp>::value,
          bool _IsVoid     = is_void<_Tp>::value>
struct __is_array_function_or_void
{
  enum
  {
    value = 0
  };
};
template <class _Tp>
struct __is_array_function_or_void<_Tp, true, false, false>
{
  enum
  {
    value = 1
  };
};
template <class _Tp>
struct __is_array_function_or_void<_Tp, false, true, false>
{
  enum
  {
    value = 2
  };
};
template <class _Tp>
struct __is_array_function_or_void<_Tp, false, false, true>
{
  enum
  {
    value = 3
  };
};
} // namespace __is_convertible_imp

template <class _Tp,
          unsigned = __is_convertible_imp::__is_array_function_or_void<__libcpp_remove_reference_t<_Tp>>::value>
struct __is_convertible_check
{
  static const size_t __v = 0;
};

template <class _Tp>
struct __is_convertible_check<_Tp, 0>
{
  static const size_t __v = sizeof(_Tp);
};

template <class _T1,
          class _T2,
          unsigned _T1_is_array_function_or_void = __is_convertible_imp::__is_array_function_or_void<_T1>::value,
          unsigned _T2_is_array_function_or_void = __is_convertible_imp::__is_array_function_or_void<_T2>::value>
struct __is_convertible_fallback
    : public integral_constant<bool, __is_convertible_imp::__is_convertible_test<_T1, _T2>::value>
{};

template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 0, 1> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 1, 1> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 2, 1> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 3, 1> : public false_type
{};

template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 0, 2> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 1, 2> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 2, 2> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 3, 2> : public false_type
{};

template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 0, 3> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 1, 3> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 2, 3> : public false_type
{};
template <class _T1, class _T2>
struct __is_convertible_fallback<_T1, _T2, 3, 3> : public true_type
{};

template <class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_convertible : public __is_convertible_fallback<_T1, _T2>
{
  static const size_t __complete_check1 = __is_convertible_check<_T1>::__v;
  static const size_t __complete_check2 = __is_convertible_check<_T2>::__v;
};

#  if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _From, class _To>
_LIBCUDACXX_INLINE_VAR constexpr bool is_convertible_v = is_convertible<_From, _To>::value;
#  endif

#endif // !_CCCL_BUILTIN_IS_CONVERTIBLE_TO

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CONVERTIBLE_H
