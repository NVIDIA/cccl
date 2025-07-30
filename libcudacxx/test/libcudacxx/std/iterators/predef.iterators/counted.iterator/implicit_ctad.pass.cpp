//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// counted_iterator

// Make sure that the implicitly-generated CTAD works.

#include <cuda/std/iterator>

#include "test_macros.h"

int main(int, char**)
{
  int array[] = {1, 2, 3};
  int* p      = array;
  cuda::std::counted_iterator iter(p, 3);
  static_assert(cuda::std::is_same_v<decltype(iter), cuda::std::counted_iterator<int*>>);

  return 0;
}
