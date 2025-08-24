//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// template <class I1, class I2>
// struct in_fun_result;

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"

#if !TEST_CUDA_COMPILER(NVCC, <, 12, 9) || TEST_STD_VER == 2017 // nvbug4460443
struct A
{
  __host__ __device__ explicit A(int);
};
// no implicit conversion
static_assert(
  !cuda::std::is_constructible_v<cuda::std::ranges::in_fun_result<A, A>, cuda::std::ranges::in_fun_result<int, int>>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 9) || TEST_STD_VER == 2017

struct B
{
  __host__ __device__ B(int);
};
// implicit conversion
static_assert(
  cuda::std::is_constructible_v<cuda::std::ranges::in_fun_result<B, B>, cuda::std::ranges::in_fun_result<int, int>>);
static_assert(
  cuda::std::is_constructible_v<cuda::std::ranges::in_fun_result<B, B>, cuda::std::ranges::in_fun_result<int, int>&>);
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::in_fun_result<B, B>,
                                            const cuda::std::ranges::in_fun_result<int, int>>);
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::in_fun_result<B, B>,
                                            const cuda::std::ranges::in_fun_result<int, int>&>);

#if !TEST_CUDA_COMPILER(NVCC, <, 12, 9) || TEST_STD_VER == 2017 // nvbug4460443
struct C
{
  __host__ __device__ C(int&);
};
static_assert(
  !cuda::std::is_constructible_v<cuda::std::ranges::in_fun_result<C, C>, cuda::std::ranges::in_fun_result<int, int>&>);
#endif // !TEST_CUDA_COMPILER(NVCC, <, 12, 9) || TEST_STD_VER == 2017

// has to be convertible via const&
static_assert(cuda::std::is_convertible_v<cuda::std::ranges::in_fun_result<int, int>&,
                                          cuda::std::ranges::in_fun_result<long, long>>);
static_assert(cuda::std::is_convertible_v<const cuda::std::ranges::in_fun_result<int, int>&,
                                          cuda::std::ranges::in_fun_result<long, long>>);
static_assert(cuda::std::is_convertible_v<cuda::std::ranges::in_fun_result<int, int>&&,
                                          cuda::std::ranges::in_fun_result<long, long>>);
static_assert(cuda::std::is_convertible_v<const cuda::std::ranges::in_fun_result<int, int>&&,
                                          cuda::std::ranges::in_fun_result<long, long>>);

// should be move constructible
static_assert(cuda::std::is_move_constructible_v<cuda::std::ranges::in_fun_result<MoveOnly, int>>);
static_assert(cuda::std::is_move_constructible_v<cuda::std::ranges::in_fun_result<int, MoveOnly>>);

// should not copy constructible with move-only type
static_assert(!cuda::std::is_copy_constructible_v<cuda::std::ranges::in_fun_result<MoveOnly, int>>);
static_assert(!cuda::std::is_copy_constructible_v<cuda::std::ranges::in_fun_result<int, MoveOnly>>);

struct NotConvertible
{};
// conversions should not work if there is no conversion
static_assert(!cuda::std::is_convertible_v<cuda::std::ranges::in_fun_result<NotConvertible, int>,
                                           cuda::std::ranges::in_fun_result<int, int>>);
static_assert(!cuda::std::is_convertible_v<cuda::std::ranges::in_fun_result<int, NotConvertible>,
                                           cuda::std::ranges::in_fun_result<int, int>>);

template <class T>
struct ConvertibleFrom
{
  __host__ __device__ constexpr ConvertibleFrom(T c)
      : content{c}
  {}
  T content;
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::in_fun_result<int, double> res{10, 0.};
    assert(res.in == 10);
    assert(res.fun == 0.);
    cuda::std::ranges::in_fun_result<ConvertibleFrom<int>, ConvertibleFrom<double>> res2 = res;
    assert(res2.in.content == 10);
    assert(res2.fun.content == 0.);
  }
  {
    cuda::std::ranges::in_fun_result<MoveOnly, int> res{MoveOnly{}, 2};
    assert(res.in.get() == 1);
    assert(res.fun == 2);
    auto res2 = static_cast<cuda::std::ranges::in_fun_result<MoveOnly, int>>(cuda::std::move(res));
    assert(res.in.get() == 0);
    assert(res2.in.get() == 1);
    assert(res2.fun == 2);
  }
  {
    auto [in, fun] = cuda::std::ranges::in_fun_result<int, int>{1, 2};
    assert(in == 1);
    assert(fun == 2);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
