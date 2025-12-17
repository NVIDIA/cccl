//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___UTILITY_POD_TUPLE_H
#define __CUDA_STD___UTILITY_POD_TUPLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/undefined.h>

/**
 * @file pod_tuple.h
 * @brief Provides a lightweight implementation of a tuple-like structure that can be
 * aggregate-initialized. It can be used to return a tuple of immovable types from a function.
 * It is guaranteed to be a structural type up to 8 elements.
 *
 * This header defines the `__tuple` template and related utilities for creating and
 * manipulating tuples with compile-time optimizations.
 *
 * @details
 * The `__tuple` structure is designed to minimize template instantiations and improve
 * compile-time performance by unrolling tuples of sizes 1-8. It also provides utilities
 * for accessing tuple elements and applying callable objects to tuple contents.
 *
 * Key features:
 * - Lightweight tuple implementation that can be aggregate initialized.
 * - Tuple elements can be direct-initialized.
 * - Compile-time optimizations for small tuples (sizes 1-8).
 * - Support for callable application via `__apply`.
 * - Utilities for accessing tuple elements using `__get`.
 */

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_COMPILER(GCC) || _CCCL_COMPILER(NVHPC)
// GCC (as of v14) does not implement the resolution of CWG1835
// https://cplusplus.github.io/CWG/issues/1835.html
// See: https://godbolt.org/z/TzxrhK6ea
#  define _CCCL_NO_CWG1835
#endif

#ifdef _CCCL_NO_CWG1835
#  define _CCCL_CWG1835_TEMPLATE
#else
#  define _CCCL_CWG1835_TEMPLATE template
#endif

template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_DECLSPEC_EMPTY_BASES __tuple;

namespace __detail
{
template <class _Fn, class _Tuple, class... _Us>
extern __undefined<_Fn, _Tuple, _Us...> __applicable_v;

template <class _Fn, class... _Ts, class... _Us>
inline constexpr bool __applicable_v<_Fn, __tuple<_Ts...>, _Us...> = __is_callable_v<_Fn, _Us..., _Ts...>;

template <class _Fn, class... _Ts, class... _Us>
inline constexpr bool __applicable_v<_Fn, __tuple<_Ts...>&, _Us...> = __is_callable_v<_Fn, _Us..., _Ts&...>;

template <class _Fn, class... _Ts, class... _Us>
inline constexpr bool __applicable_v<_Fn, const __tuple<_Ts...>&, _Us...> = __is_callable_v<_Fn, _Us..., const _Ts&...>;

template <class _Fn, class _Tuple, class... _Us>
extern __undefined<_Fn, _Tuple, _Us...> __nothrow_applicable_v;

template <class _Fn, class... _Ts, class... _Us>
inline constexpr bool __nothrow_applicable_v<_Fn, __tuple<_Ts...>, _Us...> =
  __is_nothrow_callable_v<_Fn, _Us..., _Ts...>;

template <class _Fn, class... _Ts, class... _Us>
inline constexpr bool __nothrow_applicable_v<_Fn, __tuple<_Ts...>&, _Us...> =
  __is_nothrow_callable_v<_Fn, _Us..., _Ts&...>;

template <class _Fn, class... _Ts, class... _Us>
inline constexpr bool __nothrow_applicable_v<_Fn, const __tuple<_Ts...>&, _Us...> =
  __is_nothrow_callable_v<_Fn, _Us..., const _Ts&...>;

template <size_t _Index, class _Ty>
struct __box
{
  _CCCL_NO_UNIQUE_ADDRESS _Ty __value;
};

template <class _Index, class... _Ts>
struct __tupl_base;

template <size_t... _Index, class... _Ts>
struct _CCCL_DECLSPEC_EMPTY_BASES __tupl_base<index_sequence<_Index...>, _Ts...> : __box<_Index, _Ts>...
{
  static constexpr size_t __size = sizeof...(_Ts);

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static constexpr auto
  __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(__nothrow_applicable_v<_Fn, _Self, _Us...>)
    -> decltype(auto)
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self)._CCCL_CWG1835_TEMPLATE __box<_Index, _Ts>::__value...);
  }
};
} // namespace __detail

template <class... _Ts>
struct __tuple : __detail::__tupl_base<index_sequence_for<_Ts...>, _Ts...>
{};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<>
{
  static constexpr size_t __size = 0;
};

template <class _Tp0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0>
{
  static constexpr size_t __size = 1;

  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
};

template <class _Tp0, class _Tp1>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1>
{
  static constexpr size_t __size = 2;

  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
};

template <class _Tp0, class _Tp1, class _Tp2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2>
{
  static constexpr size_t __size = 3;

  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
};

template <class _Tp0, class _Tp1, class _Tp2, class _Tp3>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3>
{
  static constexpr size_t __size = 4;

  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
};

template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4>
{
  static constexpr size_t __size = 5;

  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
  _CCCL_NO_UNIQUE_ADDRESS _Tp4 __val4;
};

template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4, class _Tp5>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5>
{
  static constexpr size_t __size = 6;

  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
  _CCCL_NO_UNIQUE_ADDRESS _Tp4 __val4;
  _CCCL_NO_UNIQUE_ADDRESS _Tp5 __val5;
};

template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4, class _Tp5, class _Tp6>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5, _Tp6>
{
  static constexpr size_t __size = 7;

  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
  _CCCL_NO_UNIQUE_ADDRESS _Tp4 __val4;
  _CCCL_NO_UNIQUE_ADDRESS _Tp5 __val5;
  _CCCL_NO_UNIQUE_ADDRESS _Tp6 __val6;
};

template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4, class _Tp5, class _Tp6, class _Tp7>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5, _Tp6, _Tp7>
{
  static constexpr size_t __size = 8;

  _CCCL_NO_UNIQUE_ADDRESS _Tp0 __val0;
  _CCCL_NO_UNIQUE_ADDRESS _Tp1 __val1;
  _CCCL_NO_UNIQUE_ADDRESS _Tp2 __val2;
  _CCCL_NO_UNIQUE_ADDRESS _Tp3 __val3;
  _CCCL_NO_UNIQUE_ADDRESS _Tp4 __val4;
  _CCCL_NO_UNIQUE_ADDRESS _Tp5 __val5;
  _CCCL_NO_UNIQUE_ADDRESS _Tp6 __val6;
  _CCCL_NO_UNIQUE_ADDRESS _Tp7 __val7;
};

template <class... _Ts>
_CCCL_HOST_DEVICE __tuple(_Ts...) -> __tuple<_Ts...>;

//
// __apply(fn, tuple, extra...)
//
#define _CCCL_TUPLE_GET(_Idx) , static_cast<_Tuple&&>(__tupl).__val##_Idx

_CCCL_EXEC_CHECK_DISABLE
_CCCL_TEMPLATE(class _Fn, class _Tuple, class... _Us)
_CCCL_REQUIRES(__detail::__applicable_v<_Fn, _Tuple, _Us...>)
_CCCL_TRIVIAL_API constexpr auto
__apply(_Fn&& __fn, _Tuple&& __tupl, _Us&&... __us) noexcept(__detail::__nothrow_applicable_v<_Fn, _Tuple, _Us...>)
  -> decltype(auto)
{
  constexpr size_t __size = remove_reference_t<_Tuple>::__size;

  if constexpr (__size == 0)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)...);
  }
  else if constexpr (__size == 1)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)... _CCCL_PP_REPEAT(1, _CCCL_TUPLE_GET));
  }
  else if constexpr (__size == 2)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)... _CCCL_PP_REPEAT(2, _CCCL_TUPLE_GET));
  }
  else if constexpr (__size == 3)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)... _CCCL_PP_REPEAT(3, _CCCL_TUPLE_GET));
  }
  else if constexpr (__size == 4)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)... _CCCL_PP_REPEAT(4, _CCCL_TUPLE_GET));
  }
  else if constexpr (__size == 5)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)... _CCCL_PP_REPEAT(5, _CCCL_TUPLE_GET));
  }
  else if constexpr (__size == 6)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)... _CCCL_PP_REPEAT(6, _CCCL_TUPLE_GET));
  }
  else if constexpr (__size == 7)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)... _CCCL_PP_REPEAT(7, _CCCL_TUPLE_GET));
  }
  else if constexpr (__size == 8)
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)... _CCCL_PP_REPEAT(8, _CCCL_TUPLE_GET));
  }
  else
  {
    return __tupl.__apply(static_cast<_Fn&&>(__fn), static_cast<_Tuple&&>(__tupl), static_cast<_Us&&>(__us)...);
  }
}

#undef _CCCL_TUPLE_GET

template <class _Fn, class _Tuple, class... _Us>
using __apply_result_t _CCCL_NODEBUG_ALIAS =
  decltype(::cuda::std::__apply(declval<_Fn>(), declval<_Tuple>(), declval<_Us>()...));

template <class _Fn, class _Tuple, class... _Us>
_CCCL_CONCEPT __applicable = __detail::__applicable_v<_Fn, _Tuple, _Us...>;

template <class _Fn, class _Tuple, class... _Us>
_CCCL_CONCEPT __nothrow_applicable = __detail::__nothrow_applicable_v<_Fn, _Tuple, _Us...>;

//
// __get<I>(tupl)
//
namespace __detail
{
template <size_t _Index, class _Value>
_CCCL_TRIVIAL_API constexpr auto __get(__box<_Index, _Value>&& __b) noexcept -> _Value&&
{
  return static_cast<_Value&&>(__b.__value);
}

template <size_t _Index, class _Value>
_CCCL_TRIVIAL_API constexpr auto __get(__box<_Index, _Value>& __b) noexcept -> _Value&
{
  return __b.__value;
}

template <size_t _Index, class _Value>
_CCCL_TRIVIAL_API constexpr auto __get(const __box<_Index, _Value>& __b) noexcept -> const _Value&
{
  return __b.__value;
}
} // namespace __detail

template <size_t _Index, class _Tuple>
_CCCL_TRIVIAL_API constexpr auto __get(_Tuple&& __tupl) noexcept -> auto&&
{
  constexpr auto __size = remove_reference_t<_Tuple>::__size;
  static_assert(_Index < __size, "Index out of bounds in __get");

  if constexpr (_Index == 0)
  {
    return static_cast<_Tuple&&>(__tupl).__val0;
  }
  else if constexpr (_Index == 1)
  {
    return static_cast<_Tuple&&>(__tupl).__val1;
  }
  else if constexpr (_Index == 2)
  {
    return static_cast<_Tuple&&>(__tupl).__val2;
  }
  else if constexpr (_Index == 3)
  {
    return static_cast<_Tuple&&>(__tupl).__val3;
  }
  else if constexpr (_Index == 4)
  {
    return static_cast<_Tuple&&>(__tupl).__val4;
  }
  else if constexpr (_Index == 5)
  {
    return static_cast<_Tuple&&>(__tupl).__val5;
  }
  else if constexpr (_Index == 6)
  {
    return static_cast<_Tuple&&>(__tupl).__val6;
  }
  else if constexpr (_Index == 7)
  {
    return static_cast<_Tuple&&>(__tupl).__val7;
  }
  else if constexpr (_Index == 8)
  {
    return static_cast<_Tuple&&>(__tupl).__val8;
  }
  else
  {
    return __detail::__get<_Index>(static_cast<_Tuple&&>(__tupl));
  }
}

//
// __decayed_tuple<Ts...>
//
template <class... _Ts>
using __decayed_tuple _CCCL_NODEBUG_ALIAS = __tuple<decay_t<_Ts>...>;

//
// __pair
//
template <class _First, class _Second>
struct __pair
{
  _CCCL_NO_UNIQUE_ADDRESS _First first;
  _CCCL_NO_UNIQUE_ADDRESS _Second second;
};

template <class _First, class _Second>
_CCCL_HOST_DEVICE __pair(_First, _Second) -> __pair<_First, _Second>;

//
// __tuple_size_v
//
template <class _Tuple>
extern __undefined<_Tuple> __tuple_size_v;

template <class... _Ts>
inline constexpr size_t __tuple_size_v<__tuple<_Ts...>> = sizeof...(_Ts);

template <class... _Ts>
inline constexpr size_t __tuple_size_v<const __tuple<_Ts...>> = sizeof...(_Ts);

template <class... _Ts>
inline constexpr size_t __tuple_size_v<__tuple<_Ts...>&> = sizeof...(_Ts);

template <class... _Ts>
inline constexpr size_t __tuple_size_v<const __tuple<_Ts...>&> = sizeof...(_Ts);

//
// __tuple_element_t
//
template <class _Tp>
_CCCL_API auto __remove_rvalue_ref(_Tp&&) noexcept -> _Tp;

template <size_t _Index, class _Tuple>
using __tuple_element_t _CCCL_NODEBUG_ALIAS =
  decltype(::cuda::std::__remove_rvalue_ref(::cuda::std::__get<_Index>(declval<_Tuple>())));

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA_STD___UTILITY_POD_TUPLE_H
