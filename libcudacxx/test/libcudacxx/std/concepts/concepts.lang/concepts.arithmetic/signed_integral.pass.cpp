//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T>
// concept signed_integral = // see below

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "arithmetic.h"
#include "test_macros.h"

using cuda::std::signed_integral;

template <typename T>
__host__ __device__ constexpr bool CheckSignedIntegralQualifiers()
{
  constexpr bool result = signed_integral<T>;
  static_assert(signed_integral<const T> == result, "");
  static_assert(signed_integral<volatile T> == result, "");
  static_assert(signed_integral<const volatile T> == result, "");

  static_assert(!signed_integral<T&>, "");
  static_assert(!signed_integral<const T&>, "");
  static_assert(!signed_integral<volatile T&>, "");
  static_assert(!signed_integral<const volatile T&>, "");

  static_assert(!signed_integral<T&&>, "");
  static_assert(!signed_integral<const T&&>, "");
  static_assert(!signed_integral<volatile T&&>, "");
  static_assert(!signed_integral<const volatile T&&>, "");

  static_assert(!signed_integral<T*>, "");
  static_assert(!signed_integral<const T*>, "");
  static_assert(!signed_integral<volatile T*>, "");
  static_assert(!signed_integral<const volatile T*>, "");

  static_assert(!signed_integral<T (*)()>, "");
  static_assert(!signed_integral<T (&)()>, "");
  static_assert(!signed_integral<T (&&)()>, "");

  return result;
}

// standard signed integers
static_assert(CheckSignedIntegralQualifiers<signed char>(), "");
static_assert(CheckSignedIntegralQualifiers<short>(), "");
static_assert(CheckSignedIntegralQualifiers<int>(), "");
static_assert(CheckSignedIntegralQualifiers<long>(), "");
static_assert(CheckSignedIntegralQualifiers<long long>(), "");

// bool and character *may* be signed
static_assert(CheckSignedIntegralQualifiers<wchar_t>() == cuda::std::is_signed_v<wchar_t>, "");
static_assert(CheckSignedIntegralQualifiers<bool>() == cuda::std::is_signed_v<bool>, "");
static_assert(CheckSignedIntegralQualifiers<char>() == cuda::std::is_signed_v<char>, "");
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(CheckSignedIntegralQualifiers<char8_t>() == cuda::std::is_signed_v<char8_t>, "");
#endif // TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(CheckSignedIntegralQualifiers<char16_t>() == cuda::std::is_signed_v<char16_t>, "");
static_assert(CheckSignedIntegralQualifiers<char32_t>() == cuda::std::is_signed_v<char32_t>, "");

// integers that aren't signed integrals
static_assert(!CheckSignedIntegralQualifiers<unsigned char>(), "");
static_assert(!CheckSignedIntegralQualifiers<unsigned short>(), "");
static_assert(!CheckSignedIntegralQualifiers<unsigned int>(), "");
static_assert(!CheckSignedIntegralQualifiers<unsigned long>(), "");
static_assert(!CheckSignedIntegralQualifiers<unsigned long long>(), "");

// extended integers
#ifndef TEST_HAS_NO_INT128_T
static_assert(CheckSignedIntegralQualifiers<__int128_t>(), "");
static_assert(!CheckSignedIntegralQualifiers<__uint128_t>(), "");
#endif

// types that aren't even integers shouldn't be signed integers!
static_assert(!signed_integral<void>, "");
static_assert(!CheckSignedIntegralQualifiers<float>(), "");
static_assert(!CheckSignedIntegralQualifiers<double>(), "");
static_assert(!CheckSignedIntegralQualifiers<long double>(), "");

static_assert(!CheckSignedIntegralQualifiers<ClassicEnum>(), "");
static_assert(!CheckSignedIntegralQualifiers<ScopedEnum>(), "");
static_assert(!CheckSignedIntegralQualifiers<EmptyStruct>(), "");
static_assert(!CheckSignedIntegralQualifiers<int EmptyStruct::*>(), "");
static_assert(!CheckSignedIntegralQualifiers<int (EmptyStruct::*)()>(), "");

#if TEST_STD_VER > 2017
static_assert(CheckSubsumption(0), "");
static_assert(CheckSubsumption(0U), "");
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  return 0;
}
