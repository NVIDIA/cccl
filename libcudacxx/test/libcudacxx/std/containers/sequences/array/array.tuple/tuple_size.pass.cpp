//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// tuple_size<array<T, N> >::value

#include <cuda/std/array>

#include "test_macros.h"

template <class T, cuda::std::size_t N>
__host__ __device__ void test()
{
  {
    typedef cuda::std::array<T, N> C;
    static_assert((cuda::std::tuple_size<C>::value == N), "");
  }
  {
    typedef cuda::std::array<T const, N> C;
    static_assert((cuda::std::tuple_size<C>::value == N), "");
  }
  {
    typedef cuda::std::array<T volatile, N> C;
    static_assert((cuda::std::tuple_size<C>::value == N), "");
  }
  {
    typedef cuda::std::array<T const volatile, N> C;
    static_assert((cuda::std::tuple_size<C>::value == N), "");
  }
}

int main(int, char**)
{
  test<double, 0>();
  test<double, 3>();
  test<double, 5>();

  return 0;
}
