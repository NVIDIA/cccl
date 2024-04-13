//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class T, size_t N> const T& get(const array<T, N>& a);

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  {
    cuda::std::array<double, 1> const array = {3.3};
    assert(cuda::std::get<0>(array) == 3.3);
  }
  {
    cuda::std::array<double, 2> const array = {3.3, 4.4};
    assert(cuda::std::get<0>(array) == 3.3);
    assert(cuda::std::get<1>(array) == 4.4);
  }
  {
    cuda::std::array<double, 3> const array = {3.3, 4.4, 5.5};
    assert(cuda::std::get<0>(array) == 3.3);
    assert(cuda::std::get<1>(array) == 4.4);
    assert(cuda::std::get<2>(array) == 5.5);
  }
  {
    cuda::std::array<double, 1> const array = {3.3};
    static_assert(cuda::std::is_same<double const&, decltype(cuda::std::get<0>(array))>::value, "");
    assert(cuda::std::get<0>(array) == 3.3);
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
