//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// We don't expose get_temporary_buffer()
// UNSUPPORTED: true

// Ensure allocator<void> is deprecated

#include <cuda/std/algorithm>

#include "test_macros.h"

TEST_FUNC void test()
{
  auto a = cuda::std::get_temporary_buffer<int>(1); // expected-warning
  cuda::std::return_temporary_buffer(a.first); // expected-warning
}

int main(int, char**)
{
  return 0;
}
