//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2>
// struct pair
// {
//     using first_type  = T1;
//     using second_type = T2;

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  using P = cuda::std::pair<float, short*>;
  static_assert((cuda::std::is_same<P::first_type, float>::value), "");
  static_assert((cuda::std::is_same<P::second_type, short*>::value), "");

  return 0;
}
