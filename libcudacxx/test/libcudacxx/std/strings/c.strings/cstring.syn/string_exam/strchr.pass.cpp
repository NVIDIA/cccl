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
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr void test_strchr(char* str, int c, char* expected_ret)
{
  const char* cstr = const_cast<const char*>(str);

  // Test cuda::std::strchr(char*, int) overload
  {
    const auto ret = cuda::std::strchr(str, c);
    assert(ret == expected_ret);
  }

  // Test cuda::std::strchr(const char*, int) overload
  {
    const auto ret = cuda::std::strchr(cstr, c);
    assert(ret == expected_ret);
  }
}

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::is_same_v<char*, decltype(cuda::std::strchr(cuda::std::declval<char*>(), int{}))>);
  static_assert(
    cuda::std::is_same_v<const char*, decltype(cuda::std::strchr(cuda::std::declval<const char*>(), int{}))>);

  {
    char str[]{""};
    test_strchr(str, '\0', str);
    test_strchr(str, 'a', nullptr);
  }
  {
    char str[]{"a"};
    test_strchr(str, '\0', str + 1);
    test_strchr(str, 'a', str);
    test_strchr(str, 'b', nullptr);
  }
  {
    char str[]{"aaa"};
    test_strchr(str, '\0', str + 3);
    test_strchr(str, 'a', str);
    test_strchr(str, 'b', nullptr);
  }
  {
    char str[]{"abcdabcd"};
    test_strchr(str, '\0', str + 8);
    test_strchr(str, 'a', str);
    test_strchr(str, 'b', str + 1);
    test_strchr(str, 'c', str + 2);
    test_strchr(str, 'd', str + 3);
    test_strchr(str, 'e', nullptr);
    test_strchr(str, 'f', nullptr);
  }

  // Test that searched character is converted from int to char
  {
    char str[]{"a"};
    test_strchr(str, '\0' + cuda::std::numeric_limits<unsigned char>::max() + 1, str + 1);
    test_strchr(str, 'a' + cuda::std::numeric_limits<unsigned char>::max() + 1, str);
    test_strchr(str, 'b' + cuda::std::numeric_limits<unsigned char>::max() + 1, nullptr);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
