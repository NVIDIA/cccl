//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_same

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_is_same()
{
  static_assert((cuda::std::is_same<T, U>::value), "");
  static_assert((!cuda::std::is_same<const T, U>::value), "");
  static_assert((!cuda::std::is_same<T, const U>::value), "");
  static_assert((cuda::std::is_same<const T, const U>::value), "");
#if TEST_STD_VER > 2011
  static_assert((cuda::std::is_same_v<T, U>), "");
  static_assert((!cuda::std::is_same_v<const T, U>), "");
  static_assert((!cuda::std::is_same_v<T, const U>), "");
  static_assert((cuda::std::is_same_v<const T, const U>), "");
#endif
}

template <class T, class U>
__host__ __device__ void test_is_same_ref()
{
  static_assert((cuda::std::is_same<T, U>::value), "");
  static_assert((cuda::std::is_same<const T, U>::value), "");
  static_assert((cuda::std::is_same<T, const U>::value), "");
  static_assert((cuda::std::is_same<const T, const U>::value), "");
#if TEST_STD_VER > 2011
  static_assert((cuda::std::is_same_v<T, U>), "");
  static_assert((cuda::std::is_same_v<const T, U>), "");
  static_assert((cuda::std::is_same_v<T, const U>), "");
  static_assert((cuda::std::is_same_v<const T, const U>), "");
#endif
}

template <class T, class U>
__host__ __device__ void test_is_not_same()
{
  static_assert((!cuda::std::is_same<T, U>::value), "");
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  test_is_same<int, int>();
  test_is_same<void, void>();
  test_is_same<Class, Class>();
  test_is_same<int*, int*>();
  test_is_same_ref<int&, int&>();

  test_is_not_same<int, void>();
  test_is_not_same<void, Class>();
  test_is_not_same<Class, int*>();
  test_is_not_same<int*, int&>();
  test_is_not_same<int&, int>();

  return 0;
}
