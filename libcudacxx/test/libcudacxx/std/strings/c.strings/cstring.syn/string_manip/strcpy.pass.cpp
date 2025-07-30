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

template <cuda::std::size_t... N>
__host__ __device__ constexpr bool equal_buffers(const char* lhs, const char* rhs, cuda::std::index_sequence<N...>)
{
  return ((lhs[N] == rhs[N]) && ...);
}

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_strcpy(const char* str, const char (&ref)[N])
{
  char buff[N]{};
  const auto ret = cuda::std::strcpy(buff, str);
  assert(ret == buff);
  assert(equal_buffers(buff, ref, cuda::std::make_index_sequence<N - 1>{}));
}

__host__ __device__ constexpr bool test()
{
  static_assert(
    cuda::std::is_same_v<char*,
                         decltype(cuda::std::strcpy(cuda::std::declval<char*>(), cuda::std::declval<const char*>()))>);

  test_strcpy("", "\0\0\0");
  test_strcpy("a", "a\0\0");
  test_strcpy("a\0", "a\0\0");
  test_strcpy("a\0sdf\0", "a\0\0\0\0\0");
  test_strcpy("hello", "hello\0\0\0\0");
  test_strcpy("hell\0o", "hell\0\0\0");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
