//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class F, class... Args>
// concept predicate;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::predicate;

static_assert(predicate<bool()>, "");
static_assert(predicate<bool (*)()>, "");
static_assert(predicate<bool (&)()>, "");

static_assert(!predicate<void()>, "");
static_assert(!predicate<void (*)()>, "");
static_assert(!predicate<void (&)()>, "");

struct S
{};

static_assert(!predicate<S(int), int>, "");
static_assert(!predicate<S(double), double>, "");
static_assert(predicate<int S::*, S*>, "");
static_assert(predicate<int (S::*)(), S*>, "");
static_assert(predicate<int (S::*)(), S&>, "");
static_assert(!predicate<void (S::*)(), S*>, "");
static_assert(!predicate<void (S::*)(), S&>, "");

static_assert(!predicate<bool(S)>, "");
static_assert(!predicate<bool(S)>, "");
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // unspecified MSVC bug
static_assert(!predicate<bool(S&), S>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(!predicate<bool(S&), S const&>, "");
static_assert(predicate<bool(S&), S&>, "");

struct Predicate
{
  __host__ __device__ bool operator()(int, double, char);
};
static_assert(predicate<Predicate, int, double, char>, "");
static_assert(predicate<Predicate&, int, double, char>, "");
static_assert(!predicate<const Predicate, int, double, char>, "");
static_assert(!predicate<const Predicate&, int, double, char>, "");

#if TEST_STD_VER > 2014 && !defined(TEST_COMPILER_NVRTC) // lambdas are not allowed in a constexpr expression
template <class Fun>
__host__ __device__ constexpr bool check_lambda(Fun)
{
  return predicate<Fun>;
}

static_assert(check_lambda([] {
                return cuda::std::true_type();
              }),
              "");
static_assert(check_lambda([]() -> int* {
                return nullptr;
              }),
              "");

struct boolean
{
  __host__ __device__ operator bool() const noexcept;
};
static_assert(check_lambda([] {
                return boolean();
              }),
              "");

struct explicit_bool
{
  __host__ __device__ explicit operator bool() const noexcept;
};
static_assert(!check_lambda([] {
  return explicit_bool();
}),
              "");
#endif // TEST_STD_VER > 2014

int main(int, char**)
{
  return 0;
}
