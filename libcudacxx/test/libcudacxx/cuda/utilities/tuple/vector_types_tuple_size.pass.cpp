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

#include "test_macros.h"

template <class VType, size_t Size>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  static_assert(cuda::std::tuple_size<VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<const VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<volatile VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<const volatile VType>::value == Size, "");
}

#define EXPAND_VECTOR_TYPE(Type) \
  test<Type##1, 1>();            \
  test<Type##2, 2>();            \
  test<Type##3, 3>();            \
  test<Type##4, 4>();

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  EXPAND_VECTOR_TYPE(char);
  EXPAND_VECTOR_TYPE(uchar);
  EXPAND_VECTOR_TYPE(short);
  EXPAND_VECTOR_TYPE(ushort);
  EXPAND_VECTOR_TYPE(int);
  EXPAND_VECTOR_TYPE(uint);
  EXPAND_VECTOR_TYPE(long);
  EXPAND_VECTOR_TYPE(ulong);
  EXPAND_VECTOR_TYPE(longlong);
  EXPAND_VECTOR_TYPE(ulonglong);
  EXPAND_VECTOR_TYPE(float);
  EXPAND_VECTOR_TYPE(double);

  return true;
}

__host__ __device__
#if !defined(TEST_COMPILER_MSVC)
  TEST_CONSTEXPR_CXX14
#endif // !TEST_COMPILER_MSVC
  bool
  test_dim3()
{
  test<dim3, 3>();
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
