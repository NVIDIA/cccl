//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class charT, class... Args>
// class basic_format_string<charT, type_identity_t<Args>...>
//
// constexpr basic_string_view<charT> get() const noexcept { return str; }

#include <cuda/std/__format_>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class CharT>
__host__ __device__ constexpr void test_get()
{
  using SV = cuda::std::basic_string_view<CharT>;

  static_assert(cuda::std::is_same_v<SV, decltype(cuda::std::declval<cuda::std::basic_format_string<CharT>>().get())>);
  static_assert(noexcept(cuda::std::declval<cuda::std::basic_format_string<CharT>>().get()));

  assert((cuda::std::basic_format_string<CharT>{TEST_STRLIT(CharT, "foo")}.get() == TEST_STRLIT(CharT, "foo")));
  assert((cuda::std::basic_format_string<CharT, int>{TEST_STRLIT(CharT, "{}")}.get() == TEST_STRLIT(CharT, "{}")));
  assert((cuda::std::basic_format_string<CharT, int, char>{TEST_STRLIT(CharT, "{} {:*>6}")}.get()
          == TEST_STRLIT(CharT, "{} {:*>6}")));

  // Embedded NUL character
  assert((cuda::std::basic_format_string<CharT, void*, bool>{TEST_STRLIT(CharT, "{}\0{}")}.get()
          == TEST_STRLIT(CharT, "{}\0{}")));
}

__host__ __device__ constexpr bool test()
{
  test_get<char>();
#if _CCCL_HAS_WCHAR_T()
  test_get<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
