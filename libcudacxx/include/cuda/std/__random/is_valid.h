//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_IS_VALID_H
#define _CUDA_STD___RANDOM_IS_VALID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// [rand.req.genl]/1.5 (C++26, per P4037R1):
// A template argument corresponding to a template parameter named RealType
// must be a (cv-unqualified) standard floating-point type or a member of an
// implementation-defined subset of extended floating-point types; otherwise
// the program is ill-formed.

template <class _Type>
inline constexpr bool __cccl_random_is_valid_realtype = __cccl_is_floating_point_v<_Type>;

// [rand.req.genl]/1.6 (C++26, per P4037R1):
// A template argument corresponding to a template parameter named IntType
// must be a (cv-unqualified) standard signed or unsigned integer type, an
// extended integer type whose width is in [width(char), width(long long)],
// or a member of an implementation-defined subset of integer types; otherwise
// the program is ill-formed.
//
// Note: This includes signed char and unsigned char but excludes plain char,
// bool, and char8_t/char16_t/char32_t/wchar_t (see P4037R1 section 4.2).
template <class _Type>
inline constexpr bool __cccl_random_is_valid_inttype = __cccl_is_integer_v<_Type>;

// [rand.req.genl]/1.7 (C++26, per P4037R1):
// A template argument corresponding to a template parameter named UIntType
// must be a (cv-unqualified) standard or extended unsigned integer type whose
// width is in [width(short), width(long long)], or a member of an
// implementation-defined subset of unsigned integer types; otherwise the
// program is ill-formed.
//
// Per P4037R1 section 4.5, unsigned char is supported unilaterally as part of
// the implementation-defined subset so that small engines like
// linear_congruential_engine<unsigned char, ...> are usable.
template <class _Type>
inline constexpr bool __cccl_random_is_valid_uinttype = __cccl_is_unsigned_integer_v<_Type>;

// [rand.req.urng]/3:
// A class G meets the uniform random bit generator requirements if G models
// uniform_random_bit_generator, invoke_result_t<G&> is an unsigned integer type,
// and G provides a nested typedef-name result_type that denotes the same type
// as invoke_result_t<G&>.
// (In particular, reject URNGs with signed result_types; our distributions cannot
// handle such generator types.)

template <class, class = void>
inline constexpr bool __cccl_random_is_valid_urng = false;
template <class _Gp>
inline constexpr bool __cccl_random_is_valid_urng<
  _Gp,
  enable_if_t<is_unsigned_v<typename _Gp::result_type>
              && is_same_v<decltype(::cuda::std::declval<_Gp&>()()), typename _Gp::result_type>>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_IS_VALID_H
