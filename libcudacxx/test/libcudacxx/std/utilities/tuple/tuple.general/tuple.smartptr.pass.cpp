//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// UNSUPPORTED: c++98, c++03

//  Tuples of smart pointers; based on bug #18350
//  auto_ptr doesn't have a copy constructor that takes a const &, but tuple does.

#include <cuda/std/tuple>
// cuda::std::___ptr not supported
// #include <memory>

#include "test_macros.h"

int main(int, char**)
{
  // cuda::std::___ptr not supported
  /*
    {
    cuda::std::tuple<cuda::std::unique_ptr<char>> up;
    cuda::std::tuple<cuda::std::shared_ptr<char>> sp;
    cuda::std::tuple<cuda::std::weak_ptr  <char>> wp;
    }
    {
    cuda::std::tuple<cuda::std::unique_ptr<char[]>> up;
    cuda::std::tuple<cuda::std::shared_ptr<char[]>> sp;
    cuda::std::tuple<cuda::std::weak_ptr  <char[]>> wp;
    }
  */
  // Smart pointers of type 'T[N]' are not tested here since they are not
  // supported by the standard nor by libc++'s implementation.
  // See https://reviews.llvm.org/D21320 for more information.

  return 0;
}
