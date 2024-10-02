//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// An array is a contiguous container

#include <cuda/std/array>
#include <cuda/std/cassert>
// #include <cuda/std/memory>
#include <cuda/std/utility>

#include "test_macros.h"

template <class Container>
__host__ __device__ TEST_CONSTEXPR_CXX14 void assert_contiguous(Container const& c)
{
  for (cuda::std::size_t i = 0; i < c.size(); ++i)
  {
    assert(*(c.begin() + i) == *(cuda::std::addressof(*c.begin()) + i));
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  assert_contiguous(cuda::std::array<double, 0>());
  assert_contiguous(cuda::std::array<double, 1>());
  assert_contiguous(cuda::std::array<double, 2>());
  assert_contiguous(cuda::std::array<double, 3>());

  assert_contiguous(cuda::std::array<char, 0>());
  assert_contiguous(cuda::std::array<char, 1>());
  assert_contiguous(cuda::std::array<char, 2>());
  assert_contiguous(cuda::std::array<char, 3>());

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_ADDRESSOF) // begin() & friends are constexpr in >= C++17 only
  static_assert(tests(), "");
#endif // TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
