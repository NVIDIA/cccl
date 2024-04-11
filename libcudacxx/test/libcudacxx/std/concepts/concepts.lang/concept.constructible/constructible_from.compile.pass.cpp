//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T, class... Args>
// concept constructible_from;
//    destructible<T> && is_constructible_v<T, Args...>;

#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::nullptr_t;

struct Empty
{};

struct Defaulted
{
  ~Defaulted() = default;
};
struct Deleted
{
  ~Deleted() = delete;
};

struct Noexcept
{
  __host__ __device__ ~Noexcept() noexcept;
};
struct NoexceptTrue
{
  __host__ __device__ ~NoexceptTrue() noexcept(true);
};
struct NoexceptFalse
{
  __host__ __device__ ~NoexceptFalse() noexcept(false);
};

struct Protected
{
protected:
  ~Protected() = default;
};
struct Private
{
private:
  ~Private() = default;
};

template <class T>
struct NoexceptDependant
{
  __host__ __device__ ~NoexceptDependant() noexcept(cuda::std::is_same_v<T, int>);
};

template <class T, class... Args>
__host__ __device__ void test()
{
  static_assert(cuda::std::constructible_from<T, Args...>
                  == (cuda::std::destructible<T> && cuda::std::is_constructible_v<T, Args...>),
                "");
}

__host__ __device__ void test()
{
  test<bool>();
  test<bool, bool>();

  test<char>();
  test<char, char>();
  test<char, int>();

  test<int>();
  test<int, int>();
  test<int, int, int>();

  test<double, int>();
  test<double, float>();

  NV_IF_TARGET(NV_IS_HOST, (test<double, long double>();))

  test<void>();
  test<void, bool>();
  test<void, int>();

  test<void*>();
  test<void*, nullptr_t>();

  test<int*>();
  test<int*, nullptr_t>();
  test<int[], int, int, int>();
  test<int[1]>();
  test<int[1], int>();
  test<int[1], int, int>();

  test<int (*)(int)>();
  test<int (*)(int), int>();
  test<int (*)(int), double>();
  test<int (*)(int), nullptr_t>();
  test<int (*)(int), int (*)(int)>();

  test<void (Empty::*)(const int&)>();
  test<void (Empty::*)(const int&), nullptr_t>();
  test<void (Empty::*)(const int&) const>();
  test<void (Empty::*)(const int&) const, void (Empty::*)(const int&)>();
  test<void (Empty::*)(const int&) volatile>();
  test<void (Empty::*)(const int&) volatile, void (Empty::*)(const int&) const volatile>();
  test<void (Empty::*)(const int&) const volatile>();
  test<void (Empty::*)(const int&) const volatile, double>();
  test<void (Empty::*)(const int&)&>();
  test<void (Empty::*)(const int&)&, void (Empty::*)(const int&) &&>();
  test<void (Empty::*)(const int&) &&>();
  test<void (Empty::*)(const int&)&&, void (Empty::*)(const int&)>();
#if TEST_STD_VER < 2020
  test<void (Empty::*)(const int&) throw()>();
  test<void (Empty::*)(const int&) throw(), void(Empty::*)(const int&) noexcept(true)>();
#endif // TEST_STD_VER < 2020
  test<void (Empty::*)(const int&) noexcept>();
  test<void (Empty::*)(const int&) noexcept(true)>();
  test<void (Empty::*)(const int&) noexcept(true), void (Empty::*)(const int&) noexcept(false)>();
  test<void (Empty::*)(const int&) noexcept(false)>();

  test<int&>();
  test<int&, int>();
  test<int&&>();
  test<int&&, int>();

  test<Empty>();

  test<Defaulted>();
  test<Deleted>();

  test<NoexceptTrue>();
  test<NoexceptFalse>();
  test<Noexcept>();

  test<Protected>();
  test<Private>();

  test<NoexceptDependant<int>>();
  test<NoexceptDependant<double>>();

  test<cuda::std::array<int, 1>>();
  test<cuda::std::array<int, 1>, int>();
  test<cuda::std::array<int, 1>, int, int>();
}

int main(int, char**)
{
  return 0;
}
