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
    param_type p;
    assert(p.mean() == 0);
    assert(p.stddev() == 1);
  }
  {
    using D = cuda::std::normal_distribution<>;
    typedef D::param_type param_type;
    param_type p(10);
    assert(p.mean() == 10);
    assert(p.stddev() == 1);
  }
  {
    using D = cuda::std::normal_distribution<>;
    typedef D::param_type param_type;
    param_type p(10, 5);
    assert(p.mean() == 10);
    assert(p.stddev() == 5);
  }

  return 0;
}
