//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// constexpr void advance_to(const_iterator it);

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/string_view>

#include "literal.h"

template <class CharT>
__host__ __device__ constexpr void test()
{
  constexpr const CharT* fmt = TEST_STRLIT(CharT, "abc");

  {
    cuda::std::basic_format_parse_context<CharT> context(fmt);

    context.advance_to(context.begin() + 1);
    assert(cuda::std::to_address(context.begin()) == fmt + 1);

    context.advance_to(context.begin() + 1);
    assert(cuda::std::to_address(context.begin()) == fmt + 2);

    context.advance_to(context.begin() + 1);
    assert(context.begin() == context.end());
  }
  {
    cuda::std::basic_string_view<CharT> view{fmt};
    cuda::std::basic_format_parse_context<CharT> context(view);

    context.advance_to(context.begin() + 1);
    assert(cuda::std::to_address(context.begin()) == fmt + 1);

    context.advance_to(context.begin() + 1);
    assert(cuda::std::to_address(context.begin()) == fmt + 2);

    context.advance_to(context.begin() + 1);
    assert(context.begin() == context.end());
  }
}

__host__ __device__ constexpr bool test()
{
  test<char>();
#if _CCCL_HAS_CHAR8_T()
  test<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test<char16_t>();
  test<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
