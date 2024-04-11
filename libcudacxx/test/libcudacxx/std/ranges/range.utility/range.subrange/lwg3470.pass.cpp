//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: icc

// gcc is unable to get the construction of b right
// UNSUPPORTED: gcc-7, gcc-8, gcc-9
// UNSUPPORTED: nvcc-11.1, nvcc-11.2

// class cuda::std::ranges::subrange;
//   Test the example from LWG 3470,
//   qualification conversions in __convertible_to_non_slicing

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

using gcc_needs_help_type = cuda::std::ranges::subrange<int**>;

__host__ __device__ constexpr bool test()
{
  // The example from LWG3470, using implicit conversion.
  int a[3]                                         = {1, 2, 3};
  int* b[3]                                        = {&a[2], &a[0], &a[1]};
  cuda::std::ranges::subrange<const int* const*> c = b;
  assert(c.begin() == b + 0);
  assert(c.end() == b + 3);

  // Also test CTAD and a subrange-to-subrange conversion.
  cuda::std::ranges::subrange d{b};
  static_assert(cuda::std::same_as<decltype(d), gcc_needs_help_type>);
  assert(d.begin() == b + 0);
  assert(d.end() == b + 3);

  cuda::std::ranges::subrange<const int* const*> e = d;
  assert(e.begin() == b + 0);
  assert(e.end() == b + 3);

  return true;
}

int main(int, char**)
{
  test();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(test());
#endif // TEST_COMPILER_CUDACC_BELOW_11_3

  return 0;
}
