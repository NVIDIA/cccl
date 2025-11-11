//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class normal_distribution

// bool operator=(const normal_distribution& x,
//                const normal_distribution& y);
// bool operator!(const normal_distribution& x,
//                const normal_distribution& y);

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    using D = cuda::std::normal_distribution<>;
    D d1(2.5, 4);
    D d2(2.5, 4);
    assert(d1 == d2);
  }
  {
    using D = cuda::std::normal_distribution<>;
    D d1(2.5, 4);
    D d2(2.5, 4.5);
    assert(d1 != d2);
  }

  return 0;
}
