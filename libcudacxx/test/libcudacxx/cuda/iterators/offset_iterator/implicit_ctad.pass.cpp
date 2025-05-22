//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// offset_iterator

// Make sure that the implicitly-generated CTAD works.

#include <cuda/iterator>

#include "test_macros.h"

int main(int, char**)
{
  int buffer[] = {1, 2, 3};
  {
    [[maybe_unused]] cuda::offset_iterator iter(buffer, 3);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::offset_iterator<int*, ptrdiff_t>>);
  }

  {
    [[maybe_unused]] cuda::offset_iterator iter(buffer, buffer);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::offset_iterator<int*, int*>>);
  }

  return 0;
}
