//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// class istream_iterator

// constexpr istream_iterator();

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_macros.h"

struct S
{
  S();
}; // not constexpr

int main(int, char**)
{
  {
    constexpr cuda::std::istream_iterator<S> it;
  }

  return 0;
}
