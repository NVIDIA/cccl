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

#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/integer_sequence.h>

/**
 * @file pod_tuple.h
 * @brief Provides a lightweight implementation of a tuple-like structure that can be
 * aggregate-initialized. It can be used to return a tuple of immovable types from a function.
 * It is guaranteed to be a structural type up to 8 elements.
 *
 * This header defines the `__tupl` template and related utilities for creating and
 * manipulating tuples with compile-time optimizations.
 *
 * @details
 * The `__tupl` structure is designed to minimize template instantiations and improve
 * compile-time performance by unrolling tuples of sizes 1-8. It also provides utilities
 * for accessing tuple elements and applying callable objects to tuple contents.
 *
 * Key features:
 * - Lightweight tuple implementation for POD types.
 * - Compile-time optimizations for small tuples (sizes 1-8).
 * - Support for callable application via `__apply`.
 * - Utilities for accessing tuple elements using `__cget`.
 * - Compatibility with CUDA device and host code.
 */

#if _CCCL_COMPILER(CLANG, <, 19)
// See https://github.com/llvm/llvm-project/issues/88077
#  define _CCCL_BROKEN_NO_UNIQUE_ADDRESS
#endif

#define _CCCL_API         _CCCL_HOST_DEVICE _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#define _CCCL_TRIVIAL_API _CCCL_API _CCCL_FORCEINLINE _CCCL_ARTIFICIAL _CCCL_NODEBUG

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Idx, class _Ty>
struct __box
{
  // Too many compiler bugs with [[no_unique_address]] to use it here.
  // E.g., https://github.com/llvm/llvm-project/issues/88077
#if !defined(_CCCL_BROKEN_NO_UNIQUE_ADDRESS)
  _CCCL_NO_UNIQUE_ADDRESS
#endif
  _Ty __value_;
};

template <auto _Value>
using __c _CCCL_NODEBUG_ALIAS = integral_constant<decltype(_Value), _Value>;

template <class _Idx, class... _Ts>
struct __tupl;

template <size_t... _Idx, class... _Ts>
struct __tupl<index_sequence<_Idx...>, _Ts...> : __box<_Idx, _Ts>...
{
  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) //
    noexcept(__is_nothrow_callable_v<_Fn, __copy_cvref_t<_Self, _Ts>..., _Us...>)
      -> __call_result_t<_Fn, __copy_cvref_t<_Self, _Ts>..., _Us...>
  {
    return static_cast<_Fn&&>(__fn)( //
      static_cast<_Self&&>(__self).__box<_Idx, _Ts>::__value_...,
      static_cast<_Us&&>(__us)...);
  }
};

// Unroll tuples of size 1-8 to bring down the number of template instantiations and to
// permit __tuple to be used to initialize a structured binding without resorting to the
// heavy-weight std::tuple protocol. This code was generated with the following:

/*
#define _CCCL_TUPLE_DEFINE_TPARAM(_Idx)  , class _CCCL_PP_CAT(_T, _Idx)
#define _CCCL_TUPLE_INDEX_SEQUENCE(_Idx) , _Idx
#define _CCCL_TUPLE_TPARAM(_Idx)         , _CCCL_PP_CAT(_T, _Idx)
#define _CCCL_TUPLE_DEFINE_ELEMENT(_Idx) _CCCL_PP_CAT(_T, _Idx) _CCCL_PP_CAT(__t, _Idx);
#define _CCCL_TUPLE_CVREF_TPARAM(_Idx)   , __copy_cvref_t<_Self, _CCCL_PP_CAT(_T, _Idx)>
#define _CCCL_TUPLE_ELEMENT(_Idx)        , static_cast<_Self&&>(__self)._CCCL_PP_CAT(__t, _Idx)
#define _CCCL_TUPLE_MBR_PTR(_Idx)        , __c<&__tupl::_CCCL_PP_CAT(__t, _Idx)>

#define _CCCL_DEFINE_TUPLE(_SizeSub1)                                                                           \
  template <class _T0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_DEFINE_TPARAM, 1)>                                 \
  struct __tupl<index_sequence<0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_INDEX_SEQUENCE, 1)>,                    \
                _T0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_TPARAM, 1)>                                          \
  {                                                                                                             \
    _T0 __t0;                                                                                                   \
    _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_DEFINE_ELEMENT, 1)                                                   \
                                                                                                                \
    template <class _Fn, class _Self, class... _Us>                                                             \
    _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(                  \
      __is_nothrow_callable_v<_Fn,                                                                              \
                         _Us...                                                                                 \
                         _CCCL_TUPLE_CVREF_TPARAM(0) _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_CVREF_TPARAM, 1)>)  \
      -> __call_result_t<_Fn,                                                                                   \
                         _Us...                                                                                 \
                         _CCCL_TUPLE_CVREF_TPARAM(0) _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_CVREF_TPARAM, 1)>   \
    {                                                                                                           \
      return static_cast<_Fn&&>(__fn)(                                                                          \
        static_cast<_Us&&>(__us)... _CCCL_TUPLE_ELEMENT(0) _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_ELEMENT, 1)); \
    }                                                                                                           \
                                                                                                                \
    template <size_t _Idx>                                                                                      \
    _CCCL_API static constexpr auto __get_mbr_ptr() noexcept                                                    \
    {                                                                                                           \
      using __result_t _CCCL_NODEBUG_ALIAS =                                                                    \
        __type_index_c<_Idx, __c<&__tupl::__t0> _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_MBR_PTR, 1)>;            \
      return __result_t::value;                                                                                 \
    }                                                                                                           \
  }

_CCCL_DEFINE_TUPLE(0);
_CCCL_DEFINE_TUPLE(1);
_CCCL_DEFINE_TUPLE(2);
_CCCL_DEFINE_TUPLE(3);
_CCCL_DEFINE_TUPLE(4);
_CCCL_DEFINE_TUPLE(5);
_CCCL_DEFINE_TUPLE(6);
_CCCL_DEFINE_TUPLE(7);
*/

template <class _T0>
struct __tupl<index_sequence<0>, _T0>
{
  _T0 __t0;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    __is_nothrow_callable_v<_Fn, _Us..., __copy_cvref_t<_Self, _T0>>)
    -> __call_result_t<_Fn, _Us..., __copy_cvref_t<_Self, _T0>>
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__t0);
  }

  template <size_t _Idx>
  _CCCL_API static constexpr auto __get_mbr_ptr() noexcept
  {
    using __result_t _CCCL_NODEBUG_ALIAS = __type_index_c<_Idx, __c<&__tupl::__t0>>;
    return __result_t::value;
  }
};

template <class _T0, class _T1>
struct __tupl<index_sequence<0, 1>, _T0, _T1>
{
  _T0 __t0;
  _T1 __t1;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    __is_nothrow_callable_v<_Fn, _Us..., __copy_cvref_t<_Self, _T0>, __copy_cvref_t<_Self, _T1>>)
    -> __call_result_t<_Fn, _Us..., __copy_cvref_t<_Self, _T0>, __copy_cvref_t<_Self, _T1>>
  {
    return static_cast<_Fn&&>(
      __fn)(static_cast<_Us&&>(__us)..., static_cast<_Self&&>(__self).__t0, static_cast<_Self&&>(__self).__t1);
  }

  template <size_t _Idx>
  _CCCL_API static constexpr auto __get_mbr_ptr() noexcept
  {
    using __result_t _CCCL_NODEBUG_ALIAS = __type_index_c<_Idx, __c<&__tupl::__t0>, __c<&__tupl::__t1>>;
    return __result_t::value;
  }
};

template <class _T0, class _T1, class _T2>
struct __tupl<index_sequence<0, 1, 2>, _T0, _T1, _T2>
{
  _T0 __t0;
  _T1 __t1;
  _T2 __t2;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    __is_nothrow_callable_v<_Fn,
                            _Us...,
                            __copy_cvref_t<_Self, _T0>,
                            __copy_cvref_t<_Self, _T1>,
                            __copy_cvref_t<_Self, _T2>>)
    -> __call_result_t<_Fn, _Us..., __copy_cvref_t<_Self, _T0>, __copy_cvref_t<_Self, _T1>, __copy_cvref_t<_Self, _T2>>
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__t0,
      static_cast<_Self&&>(__self).__t1,
      static_cast<_Self&&>(__self).__t2);
  }

  template <size_t _Idx>
  _CCCL_API static constexpr auto __get_mbr_ptr() noexcept
  {
    using __result_t _CCCL_NODEBUG_ALIAS =
      __type_index_c<_Idx, __c<&__tupl::__t0>, __c<&__tupl::__t1>, __c<&__tupl::__t2>>;
    return __result_t::value;
  }
};

template <class _T0, class _T1, class _T2, class _T3>
struct __tupl<index_sequence<0, 1, 2, 3>, _T0, _T1, _T2, _T3>
{
  _T0 __t0;
  _T1 __t1;
  _T2 __t2;
  _T3 __t3;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    __is_nothrow_callable_v<_Fn,
                            _Us...,
                            __copy_cvref_t<_Self, _T0>,
                            __copy_cvref_t<_Self, _T1>,
                            __copy_cvref_t<_Self, _T2>,
                            __copy_cvref_t<_Self, _T3>>)
    -> __call_result_t<_Fn,
                       _Us...,
                       __copy_cvref_t<_Self, _T0>,
                       __copy_cvref_t<_Self, _T1>,
                       __copy_cvref_t<_Self, _T2>,
                       __copy_cvref_t<_Self, _T3>>
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__t0,
      static_cast<_Self&&>(__self).__t1,
      static_cast<_Self&&>(__self).__t2,
      static_cast<_Self&&>(__self).__t3);
  }

  template <size_t _Idx>
  _CCCL_API static constexpr auto __get_mbr_ptr() noexcept
  {
    using __result_t _CCCL_NODEBUG_ALIAS =
      __type_index_c<_Idx, __c<&__tupl::__t0>, __c<&__tupl::__t1>, __c<&__tupl::__t2>, __c<&__tupl::__t3>>;
    return __result_t::value;
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4>
struct __tupl<index_sequence<0, 1, 2, 3, 4>, _T0, _T1, _T2, _T3, _T4>
{
  _T0 __t0;
  _T1 __t1;
  _T2 __t2;
  _T3 __t3;
  _T4 __t4;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    __is_nothrow_callable_v<_Fn,
                            _Us...,
                            __copy_cvref_t<_Self, _T0>,
                            __copy_cvref_t<_Self, _T1>,
                            __copy_cvref_t<_Self, _T2>,
                            __copy_cvref_t<_Self, _T3>,
                            __copy_cvref_t<_Self, _T4>>)
    -> __call_result_t<_Fn,
                       _Us...,
                       __copy_cvref_t<_Self, _T0>,
                       __copy_cvref_t<_Self, _T1>,
                       __copy_cvref_t<_Self, _T2>,
                       __copy_cvref_t<_Self, _T3>,
                       __copy_cvref_t<_Self, _T4>>
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__t0,
      static_cast<_Self&&>(__self).__t1,
      static_cast<_Self&&>(__self).__t2,
      static_cast<_Self&&>(__self).__t3,
      static_cast<_Self&&>(__self).__t4);
  }

  template <size_t _Idx>
  _CCCL_API static constexpr auto __get_mbr_ptr() noexcept
  {
    using __result_t _CCCL_NODEBUG_ALIAS =
      __type_index_c<_Idx,
                     __c<&__tupl::__t0>,
                     __c<&__tupl::__t1>,
                     __c<&__tupl::__t2>,
                     __c<&__tupl::__t3>,
                     __c<&__tupl::__t4>>;
    return __result_t::value;
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5>
struct __tupl<index_sequence<0, 1, 2, 3, 4, 5>, _T0, _T1, _T2, _T3, _T4, _T5>
{
  _T0 __t0;
  _T1 __t1;
  _T2 __t2;
  _T3 __t3;
  _T4 __t4;
  _T5 __t5;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    __is_nothrow_callable_v<_Fn,
                            _Us...,
                            __copy_cvref_t<_Self, _T0>,
                            __copy_cvref_t<_Self, _T1>,
                            __copy_cvref_t<_Self, _T2>,
                            __copy_cvref_t<_Self, _T3>,
                            __copy_cvref_t<_Self, _T4>,
                            __copy_cvref_t<_Self, _T5>>)
    -> __call_result_t<_Fn,
                       _Us...,
                       __copy_cvref_t<_Self, _T0>,
                       __copy_cvref_t<_Self, _T1>,
                       __copy_cvref_t<_Self, _T2>,
                       __copy_cvref_t<_Self, _T3>,
                       __copy_cvref_t<_Self, _T4>,
                       __copy_cvref_t<_Self, _T5>>
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__t0,
      static_cast<_Self&&>(__self).__t1,
      static_cast<_Self&&>(__self).__t2,
      static_cast<_Self&&>(__self).__t3,
      static_cast<_Self&&>(__self).__t4,
      static_cast<_Self&&>(__self).__t5);
  }

  template <size_t _Idx>
  _CCCL_API static constexpr auto __get_mbr_ptr() noexcept
  {
    using __result_t _CCCL_NODEBUG_ALIAS =
      __type_index_c<_Idx,
                     __c<&__tupl::__t0>,
                     __c<&__tupl::__t1>,
                     __c<&__tupl::__t2>,
                     __c<&__tupl::__t3>,
                     __c<&__tupl::__t4>,
                     __c<&__tupl::__t5>>;
    return __result_t::value;
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5, class _T6>
struct __tupl<index_sequence<0, 1, 2, 3, 4, 5, 6>, _T0, _T1, _T2, _T3, _T4, _T5, _T6>
{
  _T0 __t0;
  _T1 __t1;
  _T2 __t2;
  _T3 __t3;
  _T4 __t4;
  _T5 __t5;
  _T6 __t6;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    __is_nothrow_callable_v<_Fn,
                            _Us...,
                            __copy_cvref_t<_Self, _T0>,
                            __copy_cvref_t<_Self, _T1>,
                            __copy_cvref_t<_Self, _T2>,
                            __copy_cvref_t<_Self, _T3>,
                            __copy_cvref_t<_Self, _T4>,
                            __copy_cvref_t<_Self, _T5>,
                            __copy_cvref_t<_Self, _T6>>)
    -> __call_result_t<_Fn,
                       _Us...,
                       __copy_cvref_t<_Self, _T0>,
                       __copy_cvref_t<_Self, _T1>,
                       __copy_cvref_t<_Self, _T2>,
                       __copy_cvref_t<_Self, _T3>,
                       __copy_cvref_t<_Self, _T4>,
                       __copy_cvref_t<_Self, _T5>,
                       __copy_cvref_t<_Self, _T6>>
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__t0,
      static_cast<_Self&&>(__self).__t1,
      static_cast<_Self&&>(__self).__t2,
      static_cast<_Self&&>(__self).__t3,
      static_cast<_Self&&>(__self).__t4,
      static_cast<_Self&&>(__self).__t5,
      static_cast<_Self&&>(__self).__t6);
  }

  template <size_t _Idx>
  _CCCL_API static constexpr auto __get_mbr_ptr() noexcept
  {
    using __result_t _CCCL_NODEBUG_ALIAS =
      __type_index_c<_Idx,
                     __c<&__tupl::__t0>,
                     __c<&__tupl::__t1>,
                     __c<&__tupl::__t2>,
                     __c<&__tupl::__t3>,
                     __c<&__tupl::__t4>,
                     __c<&__tupl::__t5>,
                     __c<&__tupl::__t6>>;
    return __result_t::value;
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5, class _T6, class _T7>
struct __tupl<index_sequence<0, 1, 2, 3, 4, 5, 6, 7>, _T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7>
{
  _T0 __t0;
  _T1 __t1;
  _T2 __t2;
  _T3 __t3;
  _T4 __t4;
  _T5 __t5;
  _T6 __t6;
  _T7 __t7;

  template <class _Fn, class _Self, class... _Us>
  _CCCL_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(
    __is_nothrow_callable_v<_Fn,
                            _Us...,
                            __copy_cvref_t<_Self, _T0>,
                            __copy_cvref_t<_Self, _T1>,
                            __copy_cvref_t<_Self, _T2>,
                            __copy_cvref_t<_Self, _T3>,
                            __copy_cvref_t<_Self, _T4>,
                            __copy_cvref_t<_Self, _T5>,
                            __copy_cvref_t<_Self, _T6>,
                            __copy_cvref_t<_Self, _T7>>)
    -> __call_result_t<_Fn,
                       _Us...,
                       __copy_cvref_t<_Self, _T0>,
                       __copy_cvref_t<_Self, _T1>,
                       __copy_cvref_t<_Self, _T2>,
                       __copy_cvref_t<_Self, _T3>,
                       __copy_cvref_t<_Self, _T4>,
                       __copy_cvref_t<_Self, _T5>,
                       __copy_cvref_t<_Self, _T6>,
                       __copy_cvref_t<_Self, _T7>>
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__t0,
      static_cast<_Self&&>(__self).__t1,
      static_cast<_Self&&>(__self).__t2,
      static_cast<_Self&&>(__self).__t3,
      static_cast<_Self&&>(__self).__t4,
      static_cast<_Self&&>(__self).__t5,
      static_cast<_Self&&>(__self).__t6,
      static_cast<_Self&&>(__self).__t7);
  }

  template <size_t _Idx>
  _CCCL_API static constexpr auto __get_mbr_ptr() noexcept
  {
    using __result_t _CCCL_NODEBUG_ALIAS =
      __type_index_c<_Idx,
                     __c<&__tupl::__t0>,
                     __c<&__tupl::__t1>,
                     __c<&__tupl::__t2>,
                     __c<&__tupl::__t3>,
                     __c<&__tupl::__t4>,
                     __c<&__tupl::__t5>,
                     __c<&__tupl::__t6>,
                     __c<&__tupl::__t7>>;
    return __result_t::value;
  }
};

template <size_t _Idx, class _Ty>
_CCCL_TRIVIAL_API constexpr auto __cget(__box<_Idx, _Ty> const& __box) noexcept -> _Ty const&
{
  return __box.__value_;
}

template <size_t _Idx, class _Tupl, auto _MbrPtr = _Tupl::template __get_mbr_ptr<_Idx>()>
_CCCL_TRIVIAL_API constexpr auto __cget(_Tupl const& __tupl) noexcept -> decltype(auto)
{
  return __tupl.*_MbrPtr;
}

template <class... _Ts>
__tupl(_Ts...) //
  ->__tupl<make_index_sequence<sizeof...(_Ts)>, _Ts...>;

template <class _Fn, class _Tupl, class... _Us>
using __apply_result_t _CCCL_NODEBUG_ALIAS =
  decltype(declval<_Tupl>().__apply(declval<_Fn>(), declval<_Tupl>(), declval<_Us>()...));

#if _CCCL_COMPILER(MSVC)
template <class... _Ts>
struct __mk_tuple_
{
  using __indices_t _CCCL_NODEBUG_ALIAS = make_index_sequence<sizeof...(_Ts)>;
  using type _CCCL_NODEBUG_ALIAS        = __tupl<__indices_t, _Ts...>;
};

template <class... _Ts>
using __tuple _CCCL_NODEBUG_ALIAS = typename __mk_tuple_<_Ts...>::type;
#else
template <class... _Ts>
using __tuple _CCCL_NODEBUG_ALIAS = __tupl<make_index_sequence<sizeof...(_Ts)>, _Ts...>;
#endif

template <class... _Ts>
using __decayed_tuple _CCCL_NODEBUG_ALIAS = __tuple<decay_t<_Ts>...>;

template <class _First, class _Second>
struct __pair
{
  _First first;
  _Second second;
};

template <class _First, class _Second>
__pair(_First, _Second) -> __pair<_First, _Second>;

_LIBCUDACXX_END_NAMESPACE_STD

#undef _CCCL_API
#undef _CCCL_TRIVIAL_API

#endif // __CUDA_STD___UTILITY_POD_TUPLE_H
