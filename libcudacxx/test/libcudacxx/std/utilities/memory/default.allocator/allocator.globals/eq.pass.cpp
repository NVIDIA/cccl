//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:

// template <class T1, class T2>
//   constexpr bool
//   operator==(const allocator<T1>&, const allocator<T2>&) throw();
//
// template <class T1, class T2>
//   constexpr bool
//   operator!=(const allocator<T1>&, const allocator<T2>&) throw();

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  cuda::std::allocator<int> a1;
  cuda::std::allocator<int> a2;
  assert(a1 == a2);
  assert(!(a1 != a2));

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
