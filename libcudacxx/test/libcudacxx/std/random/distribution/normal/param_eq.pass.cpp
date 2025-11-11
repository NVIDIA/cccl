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
// {
//     class param_type;

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/limits>

#include "test_macros.h"

int main(int, char**)
{
  {
    using D = cuda::std::normal_distribution<>;
    typedef D::param_type param_type;
    param_type p1(0.75, .5);
    param_type p2(0.75, .5);
    assert(p1 == p2);
  }
  {
    using D = cuda::std::normal_distribution<>;
    typedef D::param_type param_type;
    param_type p1(0.75, .5);
    param_type p2(0.5, .5);
    assert(p1 != p2);
  }

  return 0;
}
