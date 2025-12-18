//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/charconv>

int main(int, char**)
{
  constexpr cuda::std::size_t buff_size = 150;
  char buff[buff_size]{};
  cuda::std::to_chars(buff, buff + buff_size, bool{true}, 10);
  return 0;
}
