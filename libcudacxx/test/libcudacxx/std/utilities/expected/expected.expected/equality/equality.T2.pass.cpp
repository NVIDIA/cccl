//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T2> friend constexpr bool operator==(const expected& x, const T2& v);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct Data
{
  int i;
  __host__ __device__ constexpr Data(int ii)
      : i(ii)
  {}

  __host__ __device__ friend constexpr bool operator==(const Data& data, int ii)
  {
    return data.i == ii;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(const Data& data, int ii)
  {
    return data.i != ii;
  }
#endif // TEST_STD_VER < 2020
};

__host__ __device__ constexpr bool test()
{
  // x.has_value()
  {
    const cuda::std::expected<Data, int> e1(cuda::std::in_place, 5);
    int i2 = 10;
    int i3 = 5;
    assert(e1 != i2);
    assert(e1 == i3);
  }

  // !x.has_value()
  {
    const cuda::std::expected<Data, int> e1(cuda::std::unexpect, 5);
    int i2 = 10;
    int i3 = 5;
    assert(e1 != i2);
    assert(e1 != i3);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
