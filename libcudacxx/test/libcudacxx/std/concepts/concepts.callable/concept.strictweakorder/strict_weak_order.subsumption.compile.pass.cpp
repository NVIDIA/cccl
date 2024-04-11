//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: gcc-8, gcc-9

// template<class F, class... Args>
// concept strict_weak_order;

#include <cuda/std/concepts>

#include "test_macros.h"
#if TEST_STD_VER > 2017

struct S1
{};
struct S2
{};

struct R
{
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
  __host__ __device__ bool operator()(S2, S1) const;
  __host__ __device__ bool operator()(S2, S2) const;
};

// clang-format off
template<class F, class T, class U>
requires cuda::std::relation<F, T, U>
__host__ __device__ constexpr bool check_strict_weak_order_subsumes_relation() {
  return false;
}

template<class F, class T, class U>
requires cuda::std::strict_weak_order<F, T, U> && true
__host__ __device__ constexpr bool check_strict_weak_order_subsumes_relation() {
  return true;
}
// clang-format on

static_assert(check_strict_weak_order_subsumes_relation<int (*)(int, int), int, int>(), "");
static_assert(check_strict_weak_order_subsumes_relation<int (*)(int, double), int, double>(), "");
static_assert(check_strict_weak_order_subsumes_relation<R, S1, S1>(), "");
static_assert(check_strict_weak_order_subsumes_relation<R, S1, S2>(), "");

// clang-format off
template<class F, class T, class U>
requires cuda::std::relation<F, T, U> && true
__host__ __device__ constexpr bool check_relation_subsumes_strict_weak_order() {
  return true;
}

template<class F, class T, class U>
requires cuda::std::strict_weak_order<F, T, U>
__host__ __device__ constexpr bool check_relation_subsumes_strict_weak_order() {
  return false;
}
// clang-format on

static_assert(check_relation_subsumes_strict_weak_order<int (*)(int, int), int, int>(), "");
static_assert(check_relation_subsumes_strict_weak_order<int (*)(int, double), int, double>(), "");
static_assert(check_relation_subsumes_strict_weak_order<R, S1, S1>(), "");
static_assert(check_relation_subsumes_strict_weak_order<R, S1, S2>(), "");

// clang-format off
template<class F, class T, class U>
requires cuda::std::strict_weak_order<F, T, T> && cuda::std::strict_weak_order<F, U, U>
__host__ __device__ constexpr bool check_strict_weak_order_subsumes_itself() {
  return false;
}

template<class F, class T, class U>
requires cuda::std::strict_weak_order<F, T, U>
__host__ __device__ constexpr bool check_strict_weak_order_subsumes_itself() {
  return true;
}
// clang-format on

static_assert(check_strict_weak_order_subsumes_itself<int (*)(int, int), int, int>(), "");
static_assert(check_strict_weak_order_subsumes_itself<R, S1, S1>(), "");

#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  return 0;
}
