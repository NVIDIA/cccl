//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// tuple_element<I, array<T, N> >::type

#include <cuda/std/array>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  {
    using Exp = T;
    using C   = cuda::std::array<T, 3>;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
  }
  {
    using Exp = T const;
    using C   = cuda::std::array<T, 3> const;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
  }
  {
    using Exp = T volatile;
    using C   = cuda::std::array<T, 3> volatile;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
  }
  {
    using Exp = T const volatile;
    using C   = cuda::std::array<T, 3> const volatile;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
  }
}

int main(int, char**)
{
  test<double>();
  test<int>();

  return 0;
}
