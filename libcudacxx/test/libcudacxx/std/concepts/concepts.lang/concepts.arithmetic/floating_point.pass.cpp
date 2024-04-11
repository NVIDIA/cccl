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
// concept floating_point = // see below

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "arithmetic.h"
#include "test_macros.h"

using cuda::std::floating_point;

template <typename T>
__host__ __device__ constexpr bool CheckFloatingPointQualifiers()
{
  constexpr bool result = floating_point<T>;
  static_assert(floating_point<const T> == result, "");
  static_assert(floating_point<volatile T> == result, "");
  static_assert(floating_point<const volatile T> == result, "");

  static_assert(!floating_point<T&>, "");
  static_assert(!floating_point<const T&>, "");
  static_assert(!floating_point<volatile T&>, "");
  static_assert(!floating_point<const volatile T&>, "");

  static_assert(!floating_point<T&&>, "");
  static_assert(!floating_point<const T&&>, "");
  static_assert(!floating_point<volatile T&&>, "");
  static_assert(!floating_point<const volatile T&&>, "");

  static_assert(!floating_point<T*>, "");
  static_assert(!floating_point<const T*>, "");
  static_assert(!floating_point<volatile T*>, "");
  static_assert(!floating_point<const volatile T*>, "");

  static_assert(!floating_point<T (*)()>, "");
  static_assert(!floating_point<T (&)()>, "");
  static_assert(!floating_point<T (&&)()>, "");

  return result;
}

// floating-point types
static_assert(CheckFloatingPointQualifiers<float>(), "");
static_assert(CheckFloatingPointQualifiers<double>(), "");
static_assert(CheckFloatingPointQualifiers<long double>(), "");

// types that aren't floating-point
static_assert(!CheckFloatingPointQualifiers<signed char>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned char>(), "");
static_assert(!CheckFloatingPointQualifiers<short>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned short>(), "");
static_assert(!CheckFloatingPointQualifiers<int>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned int>(), "");
static_assert(!CheckFloatingPointQualifiers<long>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned long>(), "");
static_assert(!CheckFloatingPointQualifiers<long long>(), "");
static_assert(!CheckFloatingPointQualifiers<unsigned long long>(), "");
static_assert(!CheckFloatingPointQualifiers<wchar_t>(), "");
static_assert(!CheckFloatingPointQualifiers<bool>(), "");
static_assert(!CheckFloatingPointQualifiers<char>(), "");
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(!CheckFloatingPointQualifiers<char8_t>(), "");
#endif // TEST_STD_VER > 2017 && defined(__cpp_char8_t)
static_assert(!CheckFloatingPointQualifiers<char16_t>(), "");
static_assert(!CheckFloatingPointQualifiers<char32_t>(), "");
static_assert(!floating_point<void>, "");

static_assert(!CheckFloatingPointQualifiers<ClassicEnum>(), "");
static_assert(!CheckFloatingPointQualifiers<ScopedEnum>(), "");
static_assert(!CheckFloatingPointQualifiers<EmptyStruct>(), "");
static_assert(!CheckFloatingPointQualifiers<int EmptyStruct::*>(), "");
static_assert(!CheckFloatingPointQualifiers<int (EmptyStruct::*)()>(), "");

int main(int, char**)
{
  return 0;
}
