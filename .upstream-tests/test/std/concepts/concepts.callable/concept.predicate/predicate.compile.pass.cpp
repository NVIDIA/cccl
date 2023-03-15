//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: windows && (c++11 || c++14 || c++17)

// template<class F, class... Args>
// concept predicate;

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

using cuda::std::predicate;

static_assert(predicate<bool()>, "");
static_assert(predicate<bool (*)()>, "");
static_assert(predicate<bool (&)()>, "");

static_assert(!predicate<void()>, "");
static_assert(!predicate<void (*)()>, "");
static_assert(!predicate<void (&)()>, "");

struct S {};

static_assert(!predicate<S(int), int>, "");
static_assert(!predicate<S(double), double>, "");
static_assert(predicate<int S::*, S*>, "");
static_assert(predicate<int (S::*)(), S*>, "");
static_assert(predicate<int (S::*)(), S&>, "");
static_assert(!predicate<void (S::*)(), S*>, "");
static_assert(!predicate<void (S::*)(), S&>, "");

static_assert(!predicate<bool(S)>, "");
static_assert(!predicate<bool(S)>, "");
static_assert(!predicate<bool(S&), S>, "");
static_assert(!predicate<bool(S&), S const&>, "");
static_assert(predicate<bool(S&), S&>, "");

struct Predicate {
  __host__ __device__ bool operator()(int, double, char);
};
static_assert(predicate<Predicate, int, double, char>, "");
static_assert(predicate<Predicate&, int, double, char>, "");
static_assert(!predicate<const Predicate, int, double, char>, "");
static_assert(!predicate<const Predicate&, int, double, char>, "");

#if TEST_STD_VER > 17
constexpr bool check_lambda(auto) { return false; }

constexpr bool check_lambda(predicate auto) { return true; }
#endif // TEST_STD_VER > 17

#if TEST_STD_VER > 14
static_assert(check_lambda([] { return std::true_type(); }), "");
static_assert(check_lambda([]() -> int* { return nullptr; }), "");

struct boolean {
  __host__ __device__ operator bool() const noexcept;
};
static_assert(check_lambda([] { return boolean(); }), "");

struct explicit_bool {
  __host__ __device__ explicit operator bool() const noexcept;
};
static_assert(!check_lambda([] { return explicit_bool(); }), "");
#endif // TEST_STD_VER > 14

int main(int, char**) { return 0; }
