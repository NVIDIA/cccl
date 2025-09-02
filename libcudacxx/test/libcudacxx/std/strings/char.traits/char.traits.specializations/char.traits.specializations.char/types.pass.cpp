//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string_>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::char_type, char>);
static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::int_type, int>);
// static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::off_type, cuda::std::streamoff>);
// static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::pos_type, cuda::std::streampos>);
// static_assert(cuda::std::is_same_v<cuda::std::char_traits<char>::state_type, cuda::std::mbstate_t>);
// static_assert(std::is_same_v<std::char_traits<char>::comparison_category, std::strong_ordering>);

int main(int, char**)
{
  return 0;
}
