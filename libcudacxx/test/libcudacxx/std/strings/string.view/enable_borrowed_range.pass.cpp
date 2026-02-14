//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// template<class charT, class traits>
// inline constexpr bool ranges::enable_borrowed_range<
//     basic_string_view<charT, traits>> = true;

#include <cuda/std/string_view>

static_assert(cuda::std::ranges::enable_borrowed_range<cuda::std::basic_string_view<char>>);
#if _CCCL_HAS_CHAR8_T()
static_assert(cuda::std::ranges::enable_borrowed_range<cuda::std::basic_string_view<char8_t>>);
#endif // _CCCL_HAS_CHAR8_T()
static_assert(cuda::std::ranges::enable_borrowed_range<cuda::std::basic_string_view<char16_t>>);
static_assert(cuda::std::ranges::enable_borrowed_range<cuda::std::basic_string_view<char32_t>>);
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::ranges::enable_borrowed_range<cuda::std::basic_string_view<wchar_t>>);
#endif // _CCCL_HAS_WCHAR_T()

int main(int, char**)
{
  return 0;
}
