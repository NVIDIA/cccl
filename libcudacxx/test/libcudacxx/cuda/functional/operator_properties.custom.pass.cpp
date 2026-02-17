//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

/***********************************************************************************************************************
 * Custom type and operator definitions
 **********************************************************************************************************************/

struct MyInt
{
  int value;
};

struct MyAdd
{};

/***********************************************************************************************************************
 * User specializations of operator properties
 **********************************************************************************************************************/

template <>
inline constexpr bool cuda::is_associative_v<MyAdd, MyInt> = true;

template <>
inline constexpr bool cuda::is_commutative_v<MyAdd, MyInt> = true;

template <>
__host__ __device__ constexpr auto cuda::identity_element<MyAdd, MyInt>() noexcept
{
  return MyInt{3};
}

template <>
__host__ __device__ constexpr auto cuda::absorbing_element<MyAdd, MyInt>() noexcept
{
  return MyInt{4};
}

/***********************************************************************************************************************
 * Test dispatch
 **********************************************************************************************************************/

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::is_associative_v<MyAdd, MyInt>);
  static_assert(cuda::is_commutative_v<MyAdd, MyInt>);
  static_assert(cuda::has_identity_element_v<MyAdd, MyInt>);
  static_assert(cuda::has_absorbing_element_v<MyAdd, MyInt>);
  assert((cuda::identity_element<MyAdd, MyInt>().value == MyInt{3}.value));
  assert((cuda::absorbing_element<MyAdd, MyInt>().value == MyInt{4}.value));
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
