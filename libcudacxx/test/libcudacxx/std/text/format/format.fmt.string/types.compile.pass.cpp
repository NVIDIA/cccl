//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

//  template<class... Args>
//    using format_string =
//      basic_format_string<char, type_identity_t<Args>...>;
//  template<class... Args>
//    using wformat_string =
//      basic_format_string<wchar_t, type_identity_t<Args>...>;

#include <cuda/std/__format_>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::std::format_string<>, cuda::std::basic_format_string<char>>);
static_assert(cuda::std::is_same_v<cuda::std::format_string<int>, cuda::std::basic_format_string<char, int>>);
static_assert(
  cuda::std::is_same_v<cuda::std::format_string<int, bool>, cuda::std::basic_format_string<char, int, bool>>);
static_assert(cuda::std::is_same_v<cuda::std::format_string<int, bool, void*>,
                                   cuda::std::basic_format_string<char, int, bool, void*>>);
#if _CCCL_HAS_WCHAR_T()
static_assert(cuda::std::is_same_v<cuda::std::wformat_string<>, cuda::std::basic_format_string<wchar_t>>);
static_assert(cuda::std::is_same_v<cuda::std::wformat_string<int>, cuda::std::basic_format_string<wchar_t, int>>);
static_assert(
  cuda::std::is_same_v<cuda::std::wformat_string<int, bool>, cuda::std::basic_format_string<wchar_t, int, bool>>);
static_assert(cuda::std::is_same_v<cuda::std::wformat_string<int, bool, void*>,
                                   cuda::std::basic_format_string<wchar_t, int, bool, void*>>);
#endif // _CCCL_HAS_WCHAR_T()

int main(int, char**)
{
  return 0;
}
