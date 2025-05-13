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

template <class VT, class T, size_t N>
__host__ __device__ constexpr void test_tuple_element()
{
  static_assert(cuda::std::is_same_v<T, cuda::std::tuple_element_t<0, VT>>);
  static_assert(cuda::std::is_same_v<T, typename std::tuple_element<0, VT>::type>);

  if constexpr (N > 1)
  {
    static_assert(cuda::std::is_same_v<T, cuda::std::tuple_element_t<1, VT>>);
    static_assert(cuda::std::is_same_v<T, typename std::tuple_element<1, VT>::type>);
  }
  if constexpr (N > 2)
  {
    static_assert(cuda::std::is_same_v<T, cuda::std::tuple_element_t<2, VT>>);
    static_assert(cuda::std::is_same_v<T, typename std::tuple_element<2, VT>::type>);
  }
  if constexpr (N > 3)
  {
    static_assert(cuda::std::is_same_v<T, cuda::std::tuple_element_t<3, VT>>);
    static_assert(cuda::std::is_same_v<T, typename std::tuple_element<3, VT>::type>);
  }
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  test_tuple_element<cuda::vector_type_t<T, 1>, T, 1>();
  test_tuple_element<cuda::vector_type_t<T, 2>, T, 2>();
  test_tuple_element<cuda::vector_type_t<T, 3>, T, 3>();
  test_tuple_element<cuda::vector_type_t<T, 4>, T, 4>();
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

  test_tuple_element<cuda::dim3, unsigned int, 3>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
