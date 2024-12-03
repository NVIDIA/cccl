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

using cuda::std::__copy_cv_t;
using cuda::std::is_same;

// Ensure that we copy the proper qualifiers
static_assert(is_same<float, __copy_cv_t<int, float>>::value, "");
static_assert(is_same<const float, __copy_cv_t<const int, float>>::value, "");
static_assert(is_same<volatile float, __copy_cv_t<volatile int, float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const volatile int, float>>::value, "");

static_assert(is_same<const float, __copy_cv_t<int, const float>>::value, "");
static_assert(is_same<const float, __copy_cv_t<const int, const float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<volatile int, const float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const volatile int, const float>>::value, "");

static_assert(is_same<volatile float, __copy_cv_t<int, volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const int, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cv_t<volatile int, volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const volatile int, volatile float>>::value, "");

static_assert(is_same<const volatile float, __copy_cv_t<int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<volatile int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const volatile int, const volatile float>>::value, "");

// Ensure that we do not copy lvalue-reference qualified types to types without reference qualifiers
static_assert(is_same<float, __copy_cv_t<int&, float>>::value, "");
static_assert(is_same<float, __copy_cv_t<const int&, float>>::value, "");
static_assert(is_same<float, __copy_cv_t<volatile int&, float>>::value, "");
static_assert(is_same<float, __copy_cv_t<const volatile int&, float>>::value, "");

static_assert(is_same<const float, __copy_cv_t<int&, const float>>::value, "");
static_assert(is_same<const float, __copy_cv_t<const int&, const float>>::value, "");
static_assert(is_same<const float, __copy_cv_t<volatile int&, const float>>::value, "");
static_assert(is_same<const float, __copy_cv_t<const volatile int&, const float>>::value, "");

static_assert(is_same<volatile float, __copy_cv_t<int&, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cv_t<const int&, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cv_t<volatile int&, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cv_t<const volatile int&, volatile float>>::value, "");

static_assert(is_same<const volatile float, __copy_cv_t<int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<volatile int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const volatile int&, const volatile float>>::value, "");

// Ensure that we do not copy from lvalue-reference qualified types to types with lvalue-reference qualifiers
static_assert(is_same<float&, __copy_cv_t<int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cv_t<const int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cv_t<volatile int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cv_t<const volatile int&, float&>>::value, "");

static_assert(is_same<const float&, __copy_cv_t<int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cv_t<const int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cv_t<volatile int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cv_t<const volatile int&, const float&>>::value, "");

static_assert(is_same<volatile float&, __copy_cv_t<int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cv_t<const int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cv_t<volatile int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cv_t<const volatile int&, volatile float&>>::value, "");

static_assert(is_same<const volatile float&, __copy_cv_t<int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cv_t<const int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cv_t<volatile int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cv_t<const volatile int&, const volatile float&>>::value, "");

// Ensure that we do not copy from lvalue-reference qualified types  to types rvalue-reference qualifiers
static_assert(is_same<float&&, __copy_cv_t<int&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cv_t<const int&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cv_t<volatile int&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cv_t<const volatile int&, float&&>>::value, "");

static_assert(is_same<const float&&, __copy_cv_t<int&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cv_t<const int&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cv_t<volatile int&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cv_t<const volatile int&, const float&&>>::value, "");

static_assert(is_same<volatile float&&, __copy_cv_t<int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cv_t<const int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cv_t<volatile int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cv_t<const volatile int&, volatile float&&>>::value, "");

static_assert(is_same<const volatile float&&, __copy_cv_t<int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cv_t<const int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cv_t<volatile int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cv_t<const volatile int&, const volatile float&&>>::value, "");

// Ensure that we do not copy from rvalue-reference qualified types to types without reference qualifiers
static_assert(is_same<float, __copy_cv_t<int&&, float>>::value, "");
static_assert(is_same<float, __copy_cv_t<const int&&, float>>::value, "");
static_assert(is_same<float, __copy_cv_t<volatile int&&, float>>::value, "");
static_assert(is_same<float, __copy_cv_t<const volatile int&&, float>>::value, "");

static_assert(is_same<const float, __copy_cv_t<int&&, const float>>::value, "");
static_assert(is_same<const float, __copy_cv_t<const int&&, const float>>::value, "");
static_assert(is_same<const float, __copy_cv_t<volatile int&&, const float>>::value, "");
static_assert(is_same<const float, __copy_cv_t<const volatile int&&, const float>>::value, "");

static_assert(is_same<volatile float, __copy_cv_t<int&&, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cv_t<const int&&, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cv_t<volatile int&&, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cv_t<const volatile int&&, volatile float>>::value, "");

static_assert(is_same<const volatile float, __copy_cv_t<int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<volatile int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cv_t<const volatile int&&, const volatile float>>::value, "");

// Ensure that we do copy from lvalue-reference qualified types to types with lvalue-reference qualifiers
static_assert(is_same<float&, __copy_cv_t<int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cv_t<const int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cv_t<volatile int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cv_t<const volatile int&&, float&>>::value, "");

static_assert(is_same<const float&, __copy_cv_t<int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cv_t<const int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cv_t<volatile int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cv_t<const volatile int&&, const float&>>::value, "");

static_assert(is_same<volatile float&, __copy_cv_t<int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cv_t<const int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cv_t<volatile int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cv_t<const volatile int&&, volatile float&>>::value, "");

static_assert(is_same<const volatile float&, __copy_cv_t<int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cv_t<const int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cv_t<volatile int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cv_t<const volatile int&&, const volatile float&>>::value, "");

// Ensure that we do not copy from rvalue-reference qualified types to types with rvalue-reference qualifiers
static_assert(is_same<float&&, __copy_cv_t<int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cv_t<const int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cv_t<volatile int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cv_t<const volatile int&&, float&&>>::value, "");

static_assert(is_same<const float&&, __copy_cv_t<int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cv_t<const int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cv_t<volatile int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cv_t<const volatile int&&, const float&&>>::value, "");

static_assert(is_same<volatile float&&, __copy_cv_t<int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cv_t<const int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cv_t<volatile int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cv_t<const volatile int&&, volatile float&&>>::value, "");

static_assert(is_same<const volatile float&&, __copy_cv_t<int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cv_t<const int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cv_t<volatile int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cv_t<const volatile int&&, const volatile float&&>>::value, "");

int main(int, char**)
{
  return 0;
}
