//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template <class T> constexpr add_const<T>& as_const(T& t) noexcept;      // C++17
// template <class T>           add_const<T>& as_const(const T&&) = delete; // C++17

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct S
{
  int i;
};
__host__ __device__ bool operator==(const S& x, const S& y)
{
  return x.i == y.i;
}
__host__ __device__ bool operator==(const volatile S& x, const volatile S& y)
{
  return x.i == y.i;
}

template <typename T>
__host__ __device__ void test(T& t)
{
  static_assert(
    cuda::std::is_const<typename cuda::std::remove_reference<decltype(cuda::std::as_const(t))>::type>::value, "");
  static_assert(
    cuda::std::is_const<typename cuda::std::remove_reference<decltype(cuda::std::as_const<T>(t))>::type>::value, "");
  static_assert(
    cuda::std::is_const<typename cuda::std::remove_reference<decltype(cuda::std::as_const<const T>(t))>::type>::value,
    "");
  static_assert(
    cuda::std::is_const<typename cuda::std::remove_reference<decltype(cuda::std::as_const<volatile T>(t))>::type>::value,
    "");
  static_assert(
    cuda::std::is_const<
      typename cuda::std::remove_reference<decltype(cuda::std::as_const<const volatile T>(t))>::type>::value,
    "");

  assert(cuda::std::as_const(t) == t);
  assert(cuda::std::as_const<T>(t) == t);
  assert(cuda::std::as_const<const T>(t) == t);
  assert(cuda::std::as_const<volatile T>(t) == t);
  assert(cuda::std::as_const<const volatile T>(t) == t);
}

int main(int, char**)
{
  int i    = 3;
  double d = 4.0;
  S s{2};
  test(i);
  test(d);
  test(s);

  return 0;
}
