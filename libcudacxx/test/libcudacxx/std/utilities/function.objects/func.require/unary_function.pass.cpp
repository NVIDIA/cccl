//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>
// REQUIRES: c++98 || c++03 || c++11 || c++14
// unary_function was removed in C++17

// unary_function

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/functional>
#include <cuda/std/type_traits>

int main(int, char**)
{
  typedef cuda::std::unary_function<int, bool> uf;
  static_assert((cuda::std::is_same<uf::argument_type, int>::value), "");
  static_assert((cuda::std::is_same<uf::result_type, bool>::value), "");

  return 0;
}
