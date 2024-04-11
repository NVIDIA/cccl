//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class VType, class BaseType, size_t Index>
using expected_type = cuda::std::is_same<typename cuda::std::tuple_element<Index, VType>::type, BaseType>;

template <class VType, class BaseType, size_t VSize, size_t Index, cuda::std::__enable_if_t<(Index < VSize), int> = 0>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  static_assert((expected_type<VType, BaseType, Index>::value), "");
  static_assert((expected_type<const VType, const BaseType, Index>::value), "");
  static_assert((expected_type<volatile VType, volatile BaseType, Index>::value), "");
  static_assert((expected_type<const volatile VType, const volatile BaseType, Index>::value), "");
}

template <class VType, class BaseType, size_t VSize, size_t Index, cuda::std::__enable_if_t<(Index >= VSize), int> = 0>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{}

template <class VType, class BaseType, size_t VSize>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  test<VType, BaseType, VSize, 0>();
  test<VType, BaseType, VSize, 1>();
  test<VType, BaseType, VSize, 2>();
  test<VType, BaseType, VSize, 3>();
}

#define EXPAND_VECTOR_TYPE(Type, BaseType) \
  test<Type##1, BaseType, 1>();            \
  test<Type##2, BaseType, 2>();            \
  test<Type##3, BaseType, 3>();            \
  test<Type##4, BaseType, 4>();

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  EXPAND_VECTOR_TYPE(char, signed char);
  EXPAND_VECTOR_TYPE(uchar, unsigned char);
  EXPAND_VECTOR_TYPE(short, short);
  EXPAND_VECTOR_TYPE(ushort, unsigned short);
  EXPAND_VECTOR_TYPE(int, int);
  EXPAND_VECTOR_TYPE(uint, unsigned int);
  EXPAND_VECTOR_TYPE(long, long);
  EXPAND_VECTOR_TYPE(ulong, unsigned long);
  EXPAND_VECTOR_TYPE(longlong, long long);
  EXPAND_VECTOR_TYPE(ulonglong, unsigned long long);
  EXPAND_VECTOR_TYPE(float, float);
  EXPAND_VECTOR_TYPE(double, double);

  return true;
}

__host__ __device__
#if !defined(TEST_COMPILER_MSVC)
  TEST_CONSTEXPR_CXX14
#endif // !TEST_COMPILER_MSVC
  bool
  test_dim3()
{
  test<dim3, unsigned int, 3, 0>();
  test<dim3, unsigned int, 3, 1>();
  test<dim3, unsigned int, 3, 2>();
  return true;
}

int main(int arg, char** argv)
{
  test();
  test_dim3();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#  if !defined(TEST_COMPILER_MSVC)
  static_assert(test_dim3(), "");
#  endif // !TEST_COMPILER_MSVC
#endif // TEST_STD_VER >= 2014

  return 0;
}
