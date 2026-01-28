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
inline constexpr MyInt cuda::identity_element_v<MyAdd, MyInt> = MyInt{0};

template <>
inline constexpr MyInt cuda::absorbing_element_v<MyAdd, MyInt> = MyInt{0};

/***********************************************************************************************************************
 * Test dispatch
 **********************************************************************************************************************/

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::is_associative_v<MyAdd, MyInt>);
  static_assert(cuda::is_commutative_v<MyAdd, MyInt>);
  static_assert(cuda::has_identity_element_v<MyAdd, MyInt>);
  static_assert(cuda::has_absorbing_element_v<MyAdd, MyInt>);
  assert((cuda::absorbing_element_v<MyAdd, MyInt>.value == MyInt{0}.value));
  assert((cuda::identity_element_v<MyAdd, MyInt>.value == MyInt{0}.value));
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
