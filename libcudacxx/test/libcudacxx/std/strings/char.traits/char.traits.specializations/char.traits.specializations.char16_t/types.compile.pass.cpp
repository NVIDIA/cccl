//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string_>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::std::char_traits<char16_t>::char_type, char16_t>);
static_assert(cuda::std::is_same_v<cuda::std::char_traits<char16_t>::int_type, cuda::std::uint_least16_t>);
// static_assert(std::is_same<std::char_traits<char16_t>::off_type, std::streamoff>::value, "");
// static_assert(std::is_same<std::char_traits<char16_t>::pos_type, std::u16streampos>::value, "");
// static_assert(std::is_same<std::char_traits<char16_t>::state_type, std::mbstate_t>::value, "");
// static_assert(std::is_same_v<std::char_traits<char16_t>::comparison_category, std::strong_ordering>);

int main(int, char**)
{
  return 0;
}
