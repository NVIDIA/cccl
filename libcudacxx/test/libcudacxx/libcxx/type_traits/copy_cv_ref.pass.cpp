//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/type_traits>

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::__copy_cvref_t;
using cuda::std::is_same;

// Ensure that we copy the proper qualifiers
static_assert(is_same<float, __copy_cvref_t<int, float>>::value, "");
static_assert(is_same<const float, __copy_cvref_t<const int, float>>::value, "");
static_assert(is_same<volatile float, __copy_cvref_t<volatile int, float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const volatile int, float>>::value, "");

static_assert(is_same<const float, __copy_cvref_t<int, const float>>::value, "");
static_assert(is_same<const float, __copy_cvref_t<const int, const float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<volatile int, const float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const volatile int, const float>>::value, "");

static_assert(is_same<volatile float, __copy_cvref_t<int, volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const int, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cvref_t<volatile int, volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const volatile int, volatile float>>::value, "");

static_assert(is_same<const volatile float, __copy_cvref_t<int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<volatile int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const volatile int, const volatile float>>::value, "");

// Ensure that we do copy lvalue-reference qualifiers to types without reference qualifiers
static_assert(is_same<float&, __copy_cvref_t<int&, float>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&, float>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&, float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, float>>::value, "");

static_assert(is_same<const float&, __copy_cvref_t<int&, const float>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&, const float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&, const float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, const float>>::value, "");

static_assert(is_same<volatile float&, __copy_cvref_t<int&, volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&, volatile float>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&, volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, volatile float>>::value, "");

static_assert(is_same<const volatile float&, __copy_cvref_t<int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, const volatile float>>::value, "");

// Ensure that we do copy lvalue-reference qualifiers to types with lvalue-reference qualifiers
static_assert(is_same<float&, __copy_cvref_t<int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<volatile int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const volatile int&, float&>>::value, "");

static_assert(is_same<const float&, __copy_cvref_t<int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<volatile int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const volatile int&, const float&>>::value, "");

static_assert(is_same<volatile float&, __copy_cvref_t<int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const volatile int&, volatile float&>>::value, "");

static_assert(is_same<const volatile float&, __copy_cvref_t<int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, const volatile float&>>::value, "");

// Ensure that we do copy lvalue-reference qualifiers to types rvalue-reference qualifiers
static_assert(is_same<float&, __copy_cvref_t<int&, float&&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const int&, float&&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<volatile int&, float&&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const volatile int&, float&&>>::value, "");

static_assert(is_same<const float&, __copy_cvref_t<int&, const float&&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&, const float&&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<volatile int&, const float&&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const volatile int&, const float&&>>::value, "");

static_assert(is_same<volatile float&, __copy_cvref_t<int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const volatile int&, volatile float&&>>::value, "");

static_assert(is_same<const volatile float&, __copy_cvref_t<int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, const volatile float&&>>::value, "");

// Ensure that we do copy rvalue-reference qualifiers to types without reference qualifiers
static_assert(is_same<float&&, __copy_cvref_t<int&&, float>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<const int&&, float>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<volatile int&&, float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, float>>::value, "");

static_assert(is_same<const float&&, __copy_cvref_t<int&&, const float>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<const int&&, const float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<volatile int&&, const float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, const float>>::value, "");

static_assert(is_same<volatile float&&, __copy_cvref_t<int&&, volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const int&&, volatile float>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<volatile int&&, volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, volatile float>>::value, "");

static_assert(is_same<const volatile float&&, __copy_cvref_t<int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<volatile int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, const volatile float>>::value, "");

// Ensure that we do not copy rvalue-reference qualifiers to types with lvalue-reference qualifiers
static_assert(is_same<float&, __copy_cvref_t<int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<volatile int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const volatile int&&, float&>>::value, "");

static_assert(is_same<const float&, __copy_cvref_t<int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<volatile int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const volatile int&&, const float&>>::value, "");

static_assert(is_same<volatile float&, __copy_cvref_t<int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const volatile int&&, volatile float&>>::value, "");

static_assert(is_same<const volatile float&, __copy_cvref_t<int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&&, const volatile float&>>::value, "");

// Ensure that we do keep rvalue-reference qualifiers to types with rvalue-reference qualifiers
static_assert(is_same<float&&, __copy_cvref_t<int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cvref_t<const int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cvref_t<volatile int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cvref_t<const volatile int&&, float&&>>::value, "");

static_assert(is_same<const float&&, __copy_cvref_t<int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<const int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<volatile int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<const volatile int&&, const float&&>>::value, "");

static_assert(is_same<volatile float&&, __copy_cvref_t<int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<const int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<volatile int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<const volatile int&&, volatile float&&>>::value, "");

static_assert(is_same<const volatile float&&, __copy_cvref_t<int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<volatile int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, const volatile float&&>>::value, "");

int main(int, char**)
{
  return 0;
}
