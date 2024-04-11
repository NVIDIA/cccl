//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// type_traits

// add_const

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_add_const_imp()
{
  ASSERT_SAME_TYPE(const U, typename cuda::std::add_const<T>::type);
#if TEST_STD_VER > 2011
  ASSERT_SAME_TYPE(const U, cuda::std::add_const_t<T>);
#endif
}

template <class T>
__host__ __device__ void test_add_const()
{
  test_add_const_imp<T, const T>();
  test_add_const_imp<const T, const T>();
  test_add_const_imp<volatile T, volatile const T>();
  test_add_const_imp<const volatile T, const volatile T>();
}

int main(int, char**)
{
  test_add_const<void>();
  test_add_const<int>();
  test_add_const<int[3]>();
  test_add_const<int&>();
  test_add_const<const int&>();
  test_add_const<int*>();
  test_add_const<const int*>();

  return 0;
}
