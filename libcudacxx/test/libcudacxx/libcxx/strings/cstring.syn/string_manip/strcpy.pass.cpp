//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string/constexpr_c_functions.h>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class T, cuda::std::size_t... N>
__host__ __device__ constexpr bool equal_buffers(const T* lhs, const T* rhs, cuda::std::index_sequence<N...>)
{
  return ((lhs[N] == rhs[N]) && ...);
}

template <class T, cuda::std::size_t N>
__host__ __device__ constexpr void test_strcpy(const T* str, const T (&ref)[N])
{
  T buff[N]{};
  const auto ret = cuda::std::__cccl_strcpy(buff, str);
  assert(ret == buff);
  assert(equal_buffers(buff, ref, cuda::std::make_index_sequence<N - 1>{}));
}

__host__ __device__ constexpr bool test()
{
  // char
  test_strcpy<char>("", "\0\0\0");
  test_strcpy<char>("a", "a\0\0");
  test_strcpy<char>("a\0", "a\0\0");
  test_strcpy<char>("a\0sdf\0", "a\0\0\0\0\0");
  test_strcpy<char>("hello", "hello\0\0\0\0");
  test_strcpy<char>("hell\0o", "hell\0\0\0");

#if _LIBCUDACXX_HAS_CHAR8_T()
  // char8_t
  test_strcpy<char8_t>(u8"", u8"\0\0\0");
  test_strcpy<char8_t>(u8"a", u8"a\0\0");
  test_strcpy<char8_t>(u8"a\0", u8"a\0\0");
  test_strcpy<char8_t>(u8"a\0sdf\0", u8"a\0\0\0\0\0");
  test_strcpy<char8_t>(u8"hello", u8"hello\0\0\0\0");
  test_strcpy<char8_t>(u8"hell\0o", u8"hell\0\0\0");
#endif // _LIBCUDACXX_HAS_CHAR8_T()

  // char16_t
  test_strcpy<char16_t>(u"", u"\0\0\0");
  test_strcpy<char16_t>(u"a", u"a\0\0");
  test_strcpy<char16_t>(u"a\0", u"a\0\0");
  test_strcpy<char16_t>(u"a\0sdf\0", u"a\0\0\0\0\0");
  test_strcpy<char16_t>(u"hello", u"hello\0\0\0\0");
  test_strcpy<char16_t>(u"hell\0o", u"hell\0\0\0");

  // char32_t
  test_strcpy<char32_t>(U"", U"\0\0\0");
  test_strcpy<char32_t>(U"a", U"a\0\0");
  test_strcpy<char32_t>(U"a\0", U"a\0\0");
  test_strcpy<char32_t>(U"a\0sdf\0", U"a\0\0\0\0\0");
  test_strcpy<char32_t>(U"hello", U"hello\0\0\0\0");
  test_strcpy<char32_t>(U"hell\0o", U"hell\0\0\0");

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
