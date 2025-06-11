//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/vector_types>

template <class VT, size_t N>
__host__ __device__ constexpr void test_tuple_size()
{
  static_assert(cuda::std::tuple_size_v<VT> == N);
  static_assert(std::tuple_size<VT>::value == N);
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  test_tuple_size<cuda::vector_type_t<T, 1>, 1>();
  test_tuple_size<cuda::vector_type_t<T, 2>, 2>();
  test_tuple_size<cuda::vector_type_t<T, 3>, 3>();
  test_tuple_size<cuda::vector_type_t<T, 4>, 4>();
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<signed short>();
  test_type<signed int>();
  test_type<signed long>();
  test_type<signed long long>();

  test_type<unsigned char>();
  test_type<unsigned short>();
  test_type<unsigned int>();
  test_type<unsigned long>();
  test_type<unsigned long long>();

  test_type<float>();
  test_type<double>();

  test_tuple_size<cuda::dim3, 3>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
