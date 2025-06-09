//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>
//   The string_views's value type must be the same as the traits's char_type

#include <cuda/std/string_view>

int main(int, char**)
{
  [[maybe_unused]] cuda::std::basic_string_view<char, cuda::std::char_traits<char16_t>> s;

  return 0;
}
