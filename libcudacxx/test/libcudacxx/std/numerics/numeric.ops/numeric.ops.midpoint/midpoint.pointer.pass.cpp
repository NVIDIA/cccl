//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <numeric>

// template <class _Tp>
// _Tp* midpoint(_Tp* __a, _Tp* __b) noexcept
//

#include <cuda/std/cassert>
#include <cuda/std/numeric>

#include "test_macros.h"

#if TEST_STD_VER >= 2014
template <typename T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void constexpr_test()
{
  constexpr T array[1000] = {};
  ASSERT_SAME_TYPE(decltype(cuda::std::midpoint(array, array)), const T*);
  ASSERT_NOEXCEPT(cuda::std::midpoint(array, array));

  static_assert(cuda::std::midpoint(array, array) == array, "");
  static_assert(cuda::std::midpoint(array, array + 1000) == array + 500, "");

  static_assert(cuda::std::midpoint(array, array + 9) == array + 4, "");
  static_assert(cuda::std::midpoint(array, array + 10) == array + 5, "");
  static_assert(cuda::std::midpoint(array, array + 11) == array + 5, "");
  static_assert(cuda::std::midpoint(array + 9, array) == array + 5, "");
  static_assert(cuda::std::midpoint(array + 10, array) == array + 5, "");
  static_assert(cuda::std::midpoint(array + 11, array) == array + 6, "");
}
#endif // TEST_STD_VER >= 2014

template <typename T>
__host__ __device__ void runtime_test()
{
  T array[1000] = {}; // we need an array to make valid pointers
  ASSERT_SAME_TYPE(decltype(cuda::std::midpoint(array, array)), T*);
  ASSERT_NOEXCEPT(cuda::std::midpoint(array, array));

  assert(cuda::std::midpoint(array, array) == array);
  assert(cuda::std::midpoint(array, array + 1000) == array + 500);

  assert(cuda::std::midpoint(array, array + 9) == array + 4);
  assert(cuda::std::midpoint(array, array + 10) == array + 5);
  assert(cuda::std::midpoint(array, array + 11) == array + 5);
  assert(cuda::std::midpoint(array + 9, array) == array + 5);
  assert(cuda::std::midpoint(array + 10, array) == array + 5);
  assert(cuda::std::midpoint(array + 11, array) == array + 6);

  // explicit instantiation
  ASSERT_SAME_TYPE(decltype(cuda::std::midpoint<T>(array, array)), T*);
  ASSERT_NOEXCEPT(cuda::std::midpoint<T>(array, array));
  assert(cuda::std::midpoint<T>(array, array) == array);
  assert(cuda::std::midpoint<T>(array, array + 1000) == array + 500);
}

template <typename T>
__host__ __device__ void pointer_test()
{
  runtime_test<T>();
  runtime_test<const T>();
  runtime_test<volatile T>();
  runtime_test<const volatile T>();

  //  The constexpr tests are always const, but we can test them anyway.
#if TEST_STD_VER >= 2014 && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  constexpr_test<T>();
  constexpr_test<const T>();

#  if !defined(TEST_COMPILER_GCC)
  constexpr_test<volatile T>();
  constexpr_test<const volatile T>();
#  endif // !TEST_COMPILER_GCC
#endif // TEST_STD_VER >= 2014 && !TEST_COMPILER_CUDACC_BELOW_11_3
}

int main(int, char**)
{
  pointer_test<char>();
  pointer_test<int>();
  pointer_test<double>();

  return 0;
}
