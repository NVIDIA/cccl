//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// rvalue_ref

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_rvalue_ref()
{
  static_assert(cuda::std::is_reference<T>::value, "");
  static_assert(!cuda::std::is_arithmetic<T>::value, "");
  static_assert(!cuda::std::is_fundamental<T>::value, "");
  static_assert(!cuda::std::is_object<T>::value, "");
  static_assert(!cuda::std::is_scalar<T>::value, "");
  static_assert(cuda::std::is_compound<T>::value, "");
  static_assert(!cuda::std::is_member_pointer<T>::value, "");
}

int main(int, char**)
{
  test_rvalue_ref<int&&>();
  test_rvalue_ref<const int&&>();

  return 0;
}
