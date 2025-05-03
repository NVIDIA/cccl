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
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/integer_sequence.h>

/**
 * @file pod_tuple.h
 * @brief Provides a lightweight implementation of a tuple-like structure that can be
 * aggregate-initialized. It can be used to return a tuple of immovable types from a function.
 * It is guaranteed to be a structural type up to 16 elements.
 *
 * This header defines the `__tuple` template and related utilities for creating and
 * manipulating tuples with compile-time optimizations.
 *
 * @details
 * The `__tuple` structure is designed to minimize template instantiations and improve
 * compile-time performance by unrolling tuples of sizes 1-16. It also provides utilities
 * for accessing tuple elements and applying callable objects to tuple contents.
 *
 * Key features:
 * - Lightweight tuple implementation that can be aggregate initialized.
 * - Compile-time optimizations for small tuples (sizes 1-16).
 * - Support for callable application via `__apply`.
 * - Utilities for accessing tuple elements using `__get`.
 */

#define _CCCL_API               _CCCL_HOST_DEVICE _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#define _CCCL_TRIVIAL_API       _CCCL_API _CCCL_FORCEINLINE _CCCL_ARTIFICIAL _CCCL_NODEBUG
#define _CCCL_TUPL_UNROLL_LIMIT _CCCL_META_UNROLL_LIMIT

// Unroll tuples of size 1-16 to bring down the number of template instantiations and to
// permit __tuple to be used to initialize a structured binding without resorting to the
// heavy-weight std::tuple protocol. This code was generated with the following macros,
// which can be found here: https://godbolt.org/z/do7ThETo7

/*
#define _CCCL_TUPLE_DEFINE_TPARAM(_Idx)  , class _CCCL_PP_CAT(_T, _Idx)
#define _CCCL_TUPLE_TPARAM(_Idx)         , _CCCL_PP_CAT(_T, _Idx)
#define _CCCL_TUPLE_DEFINE_ELEMENT(_Idx) _CCCL_NO_UNIQUE_ADDRESS _CCCL_PP_CAT(_T, _Idx) _CCCL_PP_CAT(__t, _Idx);
#define _CCCL_TUPLE_MBR_PTR(_Idx)        , &__tupl::_CCCL_PP_CAT(__t, _Idx)

#define _CCCL_DEFINE_TUPLE(_SizeSub1)                                                                           \
  template <class _T0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_DEFINE_TPARAM, 1)>                                 \
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __tupl<_T0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_TPARAM, 1)>            \
  {                                                                                                             \
    _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;                                                                           \
    _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_DEFINE_ELEMENT, 1)                                                   \
                                                                                                                \
    _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept                                                   \
    {                                                                                                           \
      return static_cast<__mbr_list<&__tupl::__t0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_MBR_PTR, 1)>*>(nullptr); \
    }                                                                                                           \
  };

_CCCL_PP_REPEAT_REVERSE(_CCCL_TUPL_UNROLL_LIMIT, _CCCL_DEFINE_TUPLE)
*/

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunknown-warning-option") // "unknown warning group '-Wc++26-extensions'"
_CCCL_DIAG_SUPPRESS_CLANG("-Wc++26-extensions") // "pack indexing is a C++26 extension"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <auto... _Mbrs>
struct __mbr_list;

template <size_t _Idx, class _Ty>
struct __box
{
  _CCCL_NO_UNIQUE_ADDRESS _Ty value;
};

template <class _Idx, class... _Ts>
struct __tupl_base;

template <size_t... _Idx, class... _Ts>
struct _CCCL_DECLSPEC_EMPTY_BASES __tupl_base<index_sequence<_Idx...>, _Ts...> : __box<_Idx, _Ts>...
{
  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__box<_Idx, _Ts>::value...>*>(nullptr);
  }
};

template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_DECLSPEC_EMPTY_BASES __tuple //
    : __tupl_base<index_sequence_for<_Ts...>, _Ts...>
{};

template <class _T0,
          class _T1,
          class _T2,
          class _T3,
          class _T4,
          class _T5,
          class _T6,
          class _T7,
          class _T8,
          class _T9,
          class _T10,
          class _T11,
          class _T12,
          class _T13,
          class _T14,
          class _T15>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
__tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9, _T10, _T11, _T12, _T13, _T14, _T15>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;
  _CCCL_NO_UNIQUE_ADDRESS _T8 __t8;
  _CCCL_NO_UNIQUE_ADDRESS _T9 __t9;
  _CCCL_NO_UNIQUE_ADDRESS _T10 __t10;
  _CCCL_NO_UNIQUE_ADDRESS _T11 __t11;
  _CCCL_NO_UNIQUE_ADDRESS _T12 __t12;
  _CCCL_NO_UNIQUE_ADDRESS _T13 __t13;
  _CCCL_NO_UNIQUE_ADDRESS _T14 __t14;
  _CCCL_NO_UNIQUE_ADDRESS _T15 __t15;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<
      &__tuple::__t0,
      &__tuple::__t1,
      &__tuple::__t2,
      &__tuple::__t3,
      &__tuple::__t4,
      &__tuple::__t5,
      &__tuple::__t6,
      &__tuple::__t7,
      &__tuple::__t8,
      &__tuple::__t9,
      &__tuple::__t10,
      &__tuple::__t11,
      &__tuple::__t12,
      &__tuple::__t13,
      &__tuple::__t14,
      &__tuple::__t15>*>(nullptr);
  }
};

template <class _T0,
          class _T1,
          class _T2,
          class _T3,
          class _T4,
          class _T5,
          class _T6,
          class _T7,
          class _T8,
          class _T9,
          class _T10,
          class _T11,
          class _T12,
          class _T13,
          class _T14>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
__tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9, _T10, _T11, _T12, _T13, _T14>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;
  _CCCL_NO_UNIQUE_ADDRESS _T8 __t8;
  _CCCL_NO_UNIQUE_ADDRESS _T9 __t9;
  _CCCL_NO_UNIQUE_ADDRESS _T10 __t10;
  _CCCL_NO_UNIQUE_ADDRESS _T11 __t11;
  _CCCL_NO_UNIQUE_ADDRESS _T12 __t12;
  _CCCL_NO_UNIQUE_ADDRESS _T13 __t13;
  _CCCL_NO_UNIQUE_ADDRESS _T14 __t14;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<
      &__tuple::__t0,
      &__tuple::__t1,
      &__tuple::__t2,
      &__tuple::__t3,
      &__tuple::__t4,
      &__tuple::__t5,
      &__tuple::__t6,
      &__tuple::__t7,
      &__tuple::__t8,
      &__tuple::__t9,
      &__tuple::__t10,
      &__tuple::__t11,
      &__tuple::__t12,
      &__tuple::__t13,
      &__tuple::__t14>*>(nullptr);
  }
};

template <class _T0,
          class _T1,
          class _T2,
          class _T3,
          class _T4,
          class _T5,
          class _T6,
          class _T7,
          class _T8,
          class _T9,
          class _T10,
          class _T11,
          class _T12,
          class _T13>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9, _T10, _T11, _T12, _T13>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;
  _CCCL_NO_UNIQUE_ADDRESS _T8 __t8;
  _CCCL_NO_UNIQUE_ADDRESS _T9 __t9;
  _CCCL_NO_UNIQUE_ADDRESS _T10 __t10;
  _CCCL_NO_UNIQUE_ADDRESS _T11 __t11;
  _CCCL_NO_UNIQUE_ADDRESS _T12 __t12;
  _CCCL_NO_UNIQUE_ADDRESS _T13 __t13;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<
      &__tuple::__t0,
      &__tuple::__t1,
      &__tuple::__t2,
      &__tuple::__t3,
      &__tuple::__t4,
      &__tuple::__t5,
      &__tuple::__t6,
      &__tuple::__t7,
      &__tuple::__t8,
      &__tuple::__t9,
      &__tuple::__t10,
      &__tuple::__t11,
      &__tuple::__t12,
      &__tuple::__t13>*>(nullptr);
  }
};

template <class _T0,
          class _T1,
          class _T2,
          class _T3,
          class _T4,
          class _T5,
          class _T6,
          class _T7,
          class _T8,
          class _T9,
          class _T10,
          class _T11,
          class _T12>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9, _T10, _T11, _T12>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;
  _CCCL_NO_UNIQUE_ADDRESS _T8 __t8;
  _CCCL_NO_UNIQUE_ADDRESS _T9 __t9;
  _CCCL_NO_UNIQUE_ADDRESS _T10 __t10;
  _CCCL_NO_UNIQUE_ADDRESS _T11 __t11;
  _CCCL_NO_UNIQUE_ADDRESS _T12 __t12;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<
      &__tuple::__t0,
      &__tuple::__t1,
      &__tuple::__t2,
      &__tuple::__t3,
      &__tuple::__t4,
      &__tuple::__t5,
      &__tuple::__t6,
      &__tuple::__t7,
      &__tuple::__t8,
      &__tuple::__t9,
      &__tuple::__t10,
      &__tuple::__t11,
      &__tuple::__t12>*>(nullptr);
  }
};

template <class _T0,
          class _T1,
          class _T2,
          class _T3,
          class _T4,
          class _T5,
          class _T6,
          class _T7,
          class _T8,
          class _T9,
          class _T10,
          class _T11>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9, _T10, _T11>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;
  _CCCL_NO_UNIQUE_ADDRESS _T8 __t8;
  _CCCL_NO_UNIQUE_ADDRESS _T9 __t9;
  _CCCL_NO_UNIQUE_ADDRESS _T10 __t10;
  _CCCL_NO_UNIQUE_ADDRESS _T11 __t11;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0,
                                  &__tuple::__t1,
                                  &__tuple::__t2,
                                  &__tuple::__t3,
                                  &__tuple::__t4,
                                  &__tuple::__t5,
                                  &__tuple::__t6,
                                  &__tuple::__t7,
                                  &__tuple::__t8,
                                  &__tuple::__t9,
                                  &__tuple::__t10,
                                  &__tuple::__t11>*>(nullptr);
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5, class _T6, class _T7, class _T8, class _T9, class _T10>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9, _T10>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;
  _CCCL_NO_UNIQUE_ADDRESS _T8 __t8;
  _CCCL_NO_UNIQUE_ADDRESS _T9 __t9;
  _CCCL_NO_UNIQUE_ADDRESS _T10 __t10;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0,
                                  &__tuple::__t1,
                                  &__tuple::__t2,
                                  &__tuple::__t3,
                                  &__tuple::__t4,
                                  &__tuple::__t5,
                                  &__tuple::__t6,
                                  &__tuple::__t7,
                                  &__tuple::__t8,
                                  &__tuple::__t9,
                                  &__tuple::__t10>*>(nullptr);
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5, class _T6, class _T7, class _T8, class _T9>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;
  _CCCL_NO_UNIQUE_ADDRESS _T8 __t8;
  _CCCL_NO_UNIQUE_ADDRESS _T9 __t9;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0,
                                  &__tuple::__t1,
                                  &__tuple::__t2,
                                  &__tuple::__t3,
                                  &__tuple::__t4,
                                  &__tuple::__t5,
                                  &__tuple::__t6,
                                  &__tuple::__t7,
                                  &__tuple::__t8,
                                  &__tuple::__t9>*>(nullptr);
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5, class _T6, class _T7, class _T8>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;
  _CCCL_NO_UNIQUE_ADDRESS _T8 __t8;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0,
                                  &__tuple::__t1,
                                  &__tuple::__t2,
                                  &__tuple::__t3,
                                  &__tuple::__t4,
                                  &__tuple::__t5,
                                  &__tuple::__t6,
                                  &__tuple::__t7,
                                  &__tuple::__t8>*>(nullptr);
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5, class _T6, class _T7>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;
  _CCCL_NO_UNIQUE_ADDRESS _T7 __t7;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0,
                                  &__tuple::__t1,
                                  &__tuple::__t2,
                                  &__tuple::__t3,
                                  &__tuple::__t4,
                                  &__tuple::__t5,
                                  &__tuple::__t6,
                                  &__tuple::__t7>*>(nullptr);
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5, class _T6>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5, _T6>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;
  _CCCL_NO_UNIQUE_ADDRESS _T6 __t6;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0,
                                  &__tuple::__t1,
                                  &__tuple::__t2,
                                  &__tuple::__t3,
                                  &__tuple::__t4,
                                  &__tuple::__t5,
                                  &__tuple::__t6>*>(nullptr);
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4, class _T5>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4, _T5>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;
  _CCCL_NO_UNIQUE_ADDRESS _T5 __t5;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<
      __mbr_list<&__tuple::__t0, &__tuple::__t1, &__tuple::__t2, &__tuple::__t3, &__tuple::__t4, &__tuple::__t5>*>(
      nullptr);
  }
};

template <class _T0, class _T1, class _T2, class _T3, class _T4>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3, _T4>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;
  _CCCL_NO_UNIQUE_ADDRESS _T4 __t4;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0, &__tuple::__t1, &__tuple::__t2, &__tuple::__t3, &__tuple::__t4>*>(
      nullptr);
  }
};

template <class _T0, class _T1, class _T2, class _T3>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2, _T3>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;
  _CCCL_NO_UNIQUE_ADDRESS _T3 __t3;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0, &__tuple::__t1, &__tuple::__t2, &__tuple::__t3>*>(nullptr);
  }
};

template <class _T0, class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1, _T2>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;
  _CCCL_NO_UNIQUE_ADDRESS _T2 __t2;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0, &__tuple::__t1, &__tuple::__t2>*>(nullptr);
  }
};

template <class _T0, class _T1>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0, _T1>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;
  _CCCL_NO_UNIQUE_ADDRESS _T1 __t1;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0, &__tuple::__t1>*>(nullptr);
  }
};

template <class _T0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<_T0>
{
  _CCCL_NO_UNIQUE_ADDRESS _T0 __t0;

  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<&__tuple::__t0>*>(nullptr);
  }
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __tuple<>
{
  _CCCL_TRIVIAL_API static constexpr auto __mbrs() noexcept
  {
    return static_cast<__mbr_list<>*>(nullptr);
  }
};

template <class... _Ts>
_CCCL_HOST_DEVICE __tuple(_Ts...) -> __tuple<_Ts...>;

//
// __get<I>(tupl)
//
template <auto _Mbr>
using __mbr_t _CCCL_NODEBUG_ALIAS = integral_constant<decltype(_Mbr), _Mbr>;

template <size_t _Idx, class _Tupl, auto... _Mbrs>
_CCCL_TRIVIAL_API constexpr auto __get_aux(_Tupl&& __tuple, __mbr_list<_Mbrs...>*) noexcept -> decltype(auto)
{
#if defined(_CCCL_NO_PACK_INDEXING)
  return (static_cast<_Tupl&&>(__tuple).*(__type_index_c<_Idx, __mbr_t<_Mbrs>...>::value));
#else
  return (static_cast<_Tupl&&>(__tuple).*(_Mbrs...[_Idx]));
#endif
}

template <size_t _Idx, class _Tupl>
_CCCL_TRIVIAL_API constexpr auto __get(_Tupl&& __tuple) noexcept
  -> decltype(_CUDA_VSTD::__get_aux<_Idx>(static_cast<_Tupl&&>(__tuple), __tuple.__mbrs()))
{
  return _CUDA_VSTD::__get_aux<_Idx>(static_cast<_Tupl&&>(__tuple), __tuple.__mbrs());
}

//
// __apply(fn, tuple, extra...)
//
template <class _Fn, class _Tupl, auto... _Mbrs, class... _Us>
_CCCL_TRIVIAL_API constexpr auto __apply_aux(_Fn&& __fn, _Tupl&& __tuple, __mbr_list<_Mbrs...>*, _Us&&... __us) noexcept(
  noexcept(static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)..., static_cast<_Tupl&&>(__tuple).*_Mbrs...)))
  -> decltype(static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)..., static_cast<_Tupl&&>(__tuple).*_Mbrs...))
{
  return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)..., static_cast<_Tupl&&>(__tuple).*_Mbrs...);
}

template <class _Fn, class _Tupl, class... _Us>
_CCCL_TRIVIAL_API constexpr auto __apply(_Fn&& __fn, _Tupl&& __tuple, _Us&&... __us) //
  noexcept(noexcept(_CUDA_VSTD::__apply_aux(
    static_cast<_Fn&&>(__fn), static_cast<_Tupl&&>(__tuple), __tuple.__mbrs(), static_cast<_Us&&>(__us)...)))
    -> decltype(_CUDA_VSTD::__apply_aux(
      static_cast<_Fn&&>(__fn), static_cast<_Tupl&&>(__tuple), __tuple.__mbrs(), static_cast<_Us&&>(__us)...))
{
  return _CUDA_VSTD::__apply_aux(
    static_cast<_Fn&&>(__fn), static_cast<_Tupl&&>(__tuple), __tuple.__mbrs(), static_cast<_Us&&>(__us)...);
}

template <class _Fn, class _Tupl, class... _Us>
using __apply_result_t _CCCL_NODEBUG_ALIAS =
  decltype(_CUDA_VSTD::__apply_aux(declval<_Fn>(), declval<_Tupl>(), declval<_Tupl>().__mbrs(), declval<_Us>()...));

template <class _Fn, class _Tupl, class... _Us>
using __nothrow_applicable_detail_t = _CUDA_VSTD::enable_if_t<noexcept(
  _CUDA_VSTD::__apply_aux(declval<_Fn>(), declval<_Tupl>(), declval<_Tupl>().__mbrs(), declval<_Us>()...))>;

template <class _Fn, class _Tupl, class... _Us>
_CCCL_CONCEPT __applicable = _CUDA_VSTD::_IsValidExpansion<__apply_result_t, _Fn, _Tupl, _Us...>::value;

template <class _Fn, class _Tupl, class... _Us>
_CCCL_CONCEPT __nothrow_applicable =
  _CUDA_VSTD::_IsValidExpansion<__nothrow_applicable_detail_t, _Fn, _Tupl, _Us...>::value;

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

//
// __tuple_element_t
//
template <size_t _Idx, class _Tuple>
using __tuple_element_t _CCCL_NODEBUG_ALIAS =
  decltype(_CUDA_VSTD::__get_aux<_Idx>(declval<_Tuple>(), declval<_Tuple>().__mbrs()));

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#undef _CCCL_TUPL_UNROLL_LIMIT
#undef _CCCL_TRIVIAL_API
#undef _CCCL_API

#endif // __CUDA_STD___UTILITY_POD_TUPLE_H
