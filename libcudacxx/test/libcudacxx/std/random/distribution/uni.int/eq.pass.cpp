//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// bool operator=(const uniform_int_distribution& x,
//                const uniform_int_distribution& y);
// bool operator!(const uniform_int_distribution& x,
//                const uniform_int_distribution& y);

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    using D = cuda::std::uniform_int_distribution<>;
    D d1(3, 8);
    D d2(3, 8);
    assert(d1 == d2);
  }
  {
    using D = cuda::std::uniform_int_distribution<>;
    D d1(3, 8);
    D d2(3, 9);
    assert(d1 != d2);
  }

  return 0;
}
