//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr void test_strlen(const char* str, cuda::std::size_t expected)
{
  const auto ret = cuda::std::strlen(str);
  assert(ret == expected);
}

__host__ __device__ constexpr bool test()
{
  static_assert(
    cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::std::strlen(cuda::std::declval<const char*>()))>);

  test_strlen("", 0);
  test_strlen("a", 1);
  test_strlen("hi", 2);
  test_strlen("asdfgh", 6);
  test_strlen("asdfasdfasdfgweyr", 17);
  test_strlen("asdfasdfasdfgweyr1239859102384", 30);

  test_strlen("\0\0", 0);
  test_strlen("a\0", 1);
  test_strlen("h\0i", 1);
  test_strlen("asdf\0g\0h", 4);
  test_strlen("asdfa\0sdfasdfg\0we\0yr", 5);
  test_strlen("asdfasdfasdfgweyr1239859102384\0\0", 30);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
