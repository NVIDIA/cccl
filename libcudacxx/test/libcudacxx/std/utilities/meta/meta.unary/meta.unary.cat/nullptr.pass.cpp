//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// nullptr_t
//  is_null_pointer

// UNSUPPORTED: c++98, c++03, c++11

#include <cuda/std/type_traits>
#include <cuda/std/cstddef> // for cuda::std::nullptr_t

#include "test_macros.h"

template <class T>
__host__ __device__ void test_nullptr_imp() {
  static_assert(!cuda::std::is_void<T>::value, "");
  static_assert(cuda::std::is_null_pointer<T>::value, "");
  static_assert(!cuda::std::is_integral<T>::value, "");
  static_assert(!cuda::std::is_floating_point<T>::value, "");
  static_assert(!cuda::std::is_array<T>::value, "");
  static_assert(!cuda::std::is_pointer<T>::value, "");
  static_assert(!cuda::std::is_lvalue_reference<T>::value, "");
  static_assert(!cuda::std::is_rvalue_reference<T>::value, "");
  static_assert(!cuda::std::is_member_object_pointer<T>::value, "");
  static_assert(!cuda::std::is_member_function_pointer<T>::value, "");
  static_assert(!cuda::std::is_enum<T>::value, "");
  static_assert(!cuda::std::is_union<T>::value, "");
  static_assert(!cuda::std::is_class<T>::value, "");
  static_assert(!cuda::std::is_function<T>::value, "");
}

template <class T>
__host__ __device__ void test_nullptr() {
  test_nullptr_imp<T>();
  test_nullptr_imp<const T>();
  test_nullptr_imp<volatile T>();
  test_nullptr_imp<const volatile T>();
}

struct incomplete_type;

int main(int, char**) {
  test_nullptr<cuda::std::nullptr_t>();

  //  LWG#2582
  static_assert(!cuda::std::is_null_pointer<incomplete_type>::value, "");
  return 0;
}
