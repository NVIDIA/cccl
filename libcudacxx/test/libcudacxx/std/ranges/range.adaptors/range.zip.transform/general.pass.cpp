//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Some basic examples of how zip_transform_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/ranges>

int main(int, char**)
{
  cuda::std::array v1 = {1, 2};
  cuda::std::array v2 = {4, 5, 6};
  auto ztv            = cuda::std::views::zip_transform(cuda::std::plus(), v1, v2);
  auto expected       = {5, 7};
  assert(cuda::std::ranges::equal(ztv, expected));
  return 0;
}
