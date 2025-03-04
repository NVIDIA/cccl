//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// floating_point

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_floating_point_imp()
{
  static_assert(!cuda::std::is_reference<T>::value, "");
  static_assert(cuda::std::is_arithmetic<T>::value, "");
  static_assert(cuda::std::is_fundamental<T>::value, "");
  static_assert(cuda::std::is_object<T>::value, "");
  static_assert(cuda::std::is_scalar<T>::value, "");
  static_assert(!cuda::std::is_compound<T>::value, "");
  static_assert(!cuda::std::is_member_pointer<T>::value, "");
}

template <class T>
__host__ __device__ void test_floating_point()
{
  test_floating_point_imp<T>();
  test_floating_point_imp<const T>();
  test_floating_point_imp<volatile T>();
  test_floating_point_imp<const volatile T>();
}

int main(int, char**)
{
  test_floating_point<float>();
  test_floating_point<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_floating_point<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  return 0;
}
