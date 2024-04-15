//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
// constexpr allocator<T>::~allocator();

#include <cuda/std/__memory_>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool test()
{
  cuda::std::allocator<T> alloc;
  unused(alloc);

  // destructor called here
  return true;
}

int main(int, char**)
{
  test<int>();
  test<void>();
#ifdef _LIBCUDACXX_VERSION // extension
  test<int const>();
#endif // _LIBCUDACXX_VERSION

  static_assert(test<int>());
  static_assert(test<void>());
#ifdef _LIBCUDACXX_VERSION // extension
  static_assert(test<int const>());
#endif // _LIBCUDACXX_VERSION

  return 0;
}
