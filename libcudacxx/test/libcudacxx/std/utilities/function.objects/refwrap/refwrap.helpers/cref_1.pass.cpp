//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<const T> cref(const T& t);

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int i                                     = 0;
  cuda::std::reference_wrapper<const int> r = cuda::std::cref(i);
  assert(&r.get() == &i);
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && !defined(TEST_COMPILER_NVRTC)
  static_assert(test());
#endif

  return 0;
}
