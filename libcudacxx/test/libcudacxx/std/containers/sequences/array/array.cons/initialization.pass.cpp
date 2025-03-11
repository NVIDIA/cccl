//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Test all the ways of initializing a cuda::std::array.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct NoDefault
{
  __host__ __device__ constexpr NoDefault(int) {}
};

// Test default initialization
// This one isn't constexpr because omitting to initialize fundamental types
// isn't valid in a constexpr context.
struct test_default_initialization
{
  template <typename T>
  __host__ __device__ void operator()() const
  {
    cuda::std::array<T, 0> a0;
    unused(a0);
    cuda::std::array<T, 1> a1;
    unused(a1);
    cuda::std::array<T, 2> a2;
    unused(a2);
    cuda::std::array<T, 3> a3;
    unused(a3);

    cuda::std::array<NoDefault, 0> nodefault;
    unused(nodefault);
  }
};

struct test_nondefault_initialization
{
  template <typename T>
  __host__ __device__ constexpr void operator()() const
  {
    // Check direct-list-initialization syntax (introduced in C++11)
    {
      {
        cuda::std::array<T, 0> a0_0{};
        unused(a0_0);
      }
      {
        cuda::std::array<T, 1> a1_0{};
        unused(a1_0);
        cuda::std::array<T, 1> a1_1{T()};
        unused(a1_1);
      }
      {
        cuda::std::array<T, 2> a2_0{};
        unused(a2_0);
        cuda::std::array<T, 2> a2_1{T()};
        unused(a2_1);
        cuda::std::array<T, 2> a2_2{T(), T()};
        unused(a2_2);
      }
      {
        cuda::std::array<T, 3> a3_0{};
        unused(a3_0);
        cuda::std::array<T, 3> a3_1{T()};
        unused(a3_1);
        cuda::std::array<T, 3> a3_2{T(), T()};
        unused(a3_2);
        cuda::std::array<T, 3> a3_3{T(), T(), T()};
        unused(a3_3);
      }

      cuda::std::array<NoDefault, 0> nodefault{};
      unused(nodefault);
    }

    // Check copy-list-initialization syntax
    {
      {
        cuda::std::array<T, 0> a0_0 = {};
        unused(a0_0);
      }
      {
        cuda::std::array<T, 1> a1_0 = {};
        unused(a1_0);
        cuda::std::array<T, 1> a1_1 = {T()};
        unused(a1_1);
      }
      {
        cuda::std::array<T, 2> a2_0 = {};
        unused(a2_0);
        cuda::std::array<T, 2> a2_1 = {T()};
        unused(a2_1);
        cuda::std::array<T, 2> a2_2 = {T(), T()};
        unused(a2_2);
      }
      {
        cuda::std::array<T, 3> a3_0 = {};
        unused(a3_0);
        cuda::std::array<T, 3> a3_1 = {T()};
        unused(a3_1);
        cuda::std::array<T, 3> a3_2 = {T(), T()};
        unused(a3_2);
        cuda::std::array<T, 3> a3_3 = {T(), T(), T()};
        unused(a3_3);
      }

      cuda::std::array<NoDefault, 0> nodefault = {};
      unused(nodefault);
    }

    // Test aggregate initialization
    {
      {
        cuda::std::array<T, 0> a0_0 = {{}};
        unused(a0_0);
      }
      {
        cuda::std::array<T, 1> a1_0 = {{}};
        unused(a1_0);
        cuda::std::array<T, 1> a1_1 = {{T()}};
        unused(a1_1);
      }
      {
        cuda::std::array<T, 2> a2_0 = {{}};
        unused(a2_0);
        cuda::std::array<T, 2> a2_1 = {{T()}};
        unused(a2_1);
        cuda::std::array<T, 2> a2_2 = {{T(), T()}};
        unused(a2_2);
      }
      {
        cuda::std::array<T, 3> a3_0 = {{}};
        unused(a3_0);
        cuda::std::array<T, 3> a3_1 = {{T()}};
        unused(a3_1);
        cuda::std::array<T, 3> a3_2 = {{T(), T()}};
        unused(a3_2);
        cuda::std::array<T, 3> a3_3 = {{T(), T(), T()}};
        unused(a3_3);
      }

      // See http://wg21.link/LWG2157
      cuda::std::array<NoDefault, 0> nodefault = {{}};
      unused(nodefault);
    }
  }
};

// Test construction from an initializer-list
__host__ __device__ constexpr bool test_initializer_list()
{
  {
    cuda::std::array<double, 3> const a3_0 = {};
    assert(a3_0[0] == double());
    assert(a3_0[1] == double());
    assert(a3_0[2] == double());
  }
  {
    cuda::std::array<double, 3> const a3_1 = {1};
    assert(a3_1[0] == double(1));
    assert(a3_1[1] == double());
    assert(a3_1[2] == double());
  }
  {
    cuda::std::array<double, 3> const a3_2 = {1, 2.2};
    assert(a3_2[0] == double(1));
    assert(a3_2[1] == 2.2);
    assert(a3_2[2] == double());
  }
  {
    cuda::std::array<double, 3> const a3_3 = {1, 2, 3.5};
    assert(a3_3[0] == double(1));
    assert(a3_3[1] == double(2));
    assert(a3_3[2] == 3.5);
  }

  return true;
}

struct Empty
{};
struct Trivial
{
  int i;
  int j;
};
struct NonTrivial
{
  __host__ __device__ constexpr NonTrivial() {}
  __host__ __device__ constexpr NonTrivial(NonTrivial const&) {}
};
struct NonEmptyNonTrivial
{
  int i;
  int j;
  __host__ __device__ constexpr NonEmptyNonTrivial()
      : i(22)
      , j(33)
  {}
  __host__ __device__ constexpr NonEmptyNonTrivial(NonEmptyNonTrivial const&)
      : i(22)
      , j(33)
  {}
};

template <typename F>
__host__ __device__ constexpr bool with_all_types()
{
  F().template operator()<char>();
  F().template operator()<int>();
  F().template operator()<long>();
  F().template operator()<float>();
  F().template operator()<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  F().template operator()<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
  F().template operator()<Empty>();
  F().template operator()<Trivial>();
  F().template operator()<NonTrivial>();
  F().template operator()<NonEmptyNonTrivial>();
  return true;
}

int main(int, char**)
{
  with_all_types<test_nondefault_initialization>();
  with_all_types<test_default_initialization>(); // not constexpr
  test_initializer_list();
  static_assert(with_all_types<test_nondefault_initialization>(), "");
  static_assert(test_initializer_list(), "");

  return 0;
}
