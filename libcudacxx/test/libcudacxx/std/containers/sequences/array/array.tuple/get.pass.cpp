//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class T, size_t N> T& get(array<T, N>& a);

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename... T>
__host__ __device__ TEST_CONSTEXPR cuda::std::array<int, sizeof...(T)> tempArray(T... args)
{
  return {args...};
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  {
    cuda::std::array<double, 1> array = {3.3};
    assert(cuda::std::get<0>(array) == 3.3);
    cuda::std::get<0>(array) = 99.1;
    assert(cuda::std::get<0>(array) == 99.1);
  }
  {
    cuda::std::array<double, 2> array = {3.3, 4.4};
    assert(cuda::std::get<0>(array) == 3.3);
    assert(cuda::std::get<1>(array) == 4.4);
    cuda::std::get<0>(array) = 99.1;
    cuda::std::get<1>(array) = 99.2;
    assert(cuda::std::get<0>(array) == 99.1);
    assert(cuda::std::get<1>(array) == 99.2);
  }
  {
    cuda::std::array<double, 3> array = {3.3, 4.4, 5.5};
    assert(cuda::std::get<0>(array) == 3.3);
    assert(cuda::std::get<1>(array) == 4.4);
    assert(cuda::std::get<2>(array) == 5.5);
    cuda::std::get<1>(array) = 99.2;
    assert(cuda::std::get<0>(array) == 3.3);
    assert(cuda::std::get<1>(array) == 99.2);
    assert(cuda::std::get<2>(array) == 5.5);
  }
  {
    cuda::std::array<double, 1> array = {3.3};
    static_assert(cuda::std::is_same<double&, decltype(cuda::std::get<0>(array))>::value, "");
    unused(array);
  }
  {
    assert(cuda::std::get<0>(tempArray(1, 2, 3)) == 1);
    assert(cuda::std::get<1>(tempArray(1, 2, 3)) == 2);
    assert(cuda::std::get<2>(tempArray(1, 2, 3)) == 3);
  }

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2014
  static_assert(tests(), "");
#endif
  return 0;
}
