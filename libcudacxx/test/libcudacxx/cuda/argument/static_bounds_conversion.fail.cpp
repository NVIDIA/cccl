//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/argument>

using arg_t = cuda::argument::immediate<unsigned char, cuda::argument::static_bounds<0, 1000>>;

[[maybe_unused]] constexpr auto invalid_highest = cuda::argument::__traits<arg_t>::highest;

int main(int, char**)
{
  return 0;
}
