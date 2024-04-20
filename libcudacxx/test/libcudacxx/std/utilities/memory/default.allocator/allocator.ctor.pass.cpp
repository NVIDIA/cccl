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
//
// template <class T>
// class allocator
// {
// public: // All of these are constexpr after C++17
//  allocator() noexcept;
//  allocator(const allocator&) noexcept;
//  template<class U> allocator(const allocator<U>&) noexcept;
// ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/cstddef>

#include "test_macros.h"

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  typedef cuda::std::allocator<T> A1;
  typedef cuda::std::allocator<long> A2;

  A1 a1;
  A1 a1_copy = a1;
  unused(a1_copy);
  A2 a2 = a1;
  unused(a2);

  return true;
}

int main(int, char**)
{
  test<char>();
  test<char const>();
  test<void>();

#if TEST_STD_VER >= 2020
  static_assert(test<char>());
  static_assert(test<char const>());
  static_assert(test<void>());
#endif // TEST_STD_VER >= 2020
  return 0;
}
