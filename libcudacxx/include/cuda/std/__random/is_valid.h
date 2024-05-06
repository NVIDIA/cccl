//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANDOM_IS_VALID_H
#define _LIBCUDACXX___RANDOM_IS_VALID_H

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
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [rand.req.genl]/1.4:
// The effect of instantiating a template that has a template type parameter
// named RealType is undefined unless the corresponding template argument is
// cv-unqualified and is one of float, double, or long double.

template <class>
struct __libcpp_random_is_valid_realtype : false_type
{};
template <>
struct __libcpp_random_is_valid_realtype<float> : true_type
{};
template <>
struct __libcpp_random_is_valid_realtype<double> : true_type
{};
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
template <>
struct __libcpp_random_is_valid_realtype<long double> : true_type
{};
#endif // _LIBCUDACXX_HAS_NO_LONG_DOUBLE

// [rand.req.genl]/1.5:
// The effect of instantiating a template that has a template type parameter
// named IntType is undefined unless the corresponding template argument is
// cv-unqualified and is one of short, int, long, long long, unsigned short,
// unsigned int, unsigned long, or unsigned long long.

template <class>
struct __libcpp_random_is_valid_inttype : false_type
{};
template <>
struct __libcpp_random_is_valid_inttype<int8_t> : true_type
{}; // extension
template <>
struct __libcpp_random_is_valid_inttype<short> : true_type
{};
template <>
struct __libcpp_random_is_valid_inttype<int> : true_type
{};
template <>
struct __libcpp_random_is_valid_inttype<long> : true_type
{};
template <>
struct __libcpp_random_is_valid_inttype<long long> : true_type
{};
template <>
struct __libcpp_random_is_valid_inttype<uint8_t> : true_type
{}; // extension
template <>
struct __libcpp_random_is_valid_inttype<unsigned short> : true_type
{};
template <>
struct __libcpp_random_is_valid_inttype<unsigned int> : true_type
{};
template <>
struct __libcpp_random_is_valid_inttype<unsigned long> : true_type
{};
template <>
struct __libcpp_random_is_valid_inttype<unsigned long long> : true_type
{};

#ifndef _LIBCUDACXX_HAS_NO_INT128
template <>
struct __libcpp_random_is_valid_inttype<__int128_t> : true_type
{}; // extension
template <>
struct __libcpp_random_is_valid_inttype<__uint128_t> : true_type
{}; // extension
#endif // _LIBCUDACXX_HAS_NO_INT128

// [rand.req.urng]/3:
// A class G meets the uniform random bit generator requirements if G models
// uniform_random_bit_generator, invoke_result_t<G&> is an unsigned integer type,
// and G provides a nested typedef-name result_type that denotes the same type
// as invoke_result_t<G&>.
// (In particular, reject URNGs with signed result_types; our distributions cannot
// handle such generator types.)

template <class, class = void>
struct __libcpp_random_is_valid_urng : false_type
{};
template <class _Gp>
struct __libcpp_random_is_valid_urng<
  _Gp,
  enable_if_t<_CCCL_TRAIT(is_unsigned, typename _Gp::result_type)
              && _IsSame<decltype(_CUDA_VSTD::declval<_Gp&>()()), typename _Gp::result_type>::value>> : true_type
{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___RANDOM_IS_VALID_H
