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
// {
//     class param_type;

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/limits>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::uniform_int_distribution<long> D;
    typedef D::param_type param_type;
    param_type p;
    assert(p.a() == 0);
    assert(p.b() == cuda::std::numeric_limits<long>::max());
  }
  {
    typedef cuda::std::uniform_int_distribution<long> D;
    typedef D::param_type param_type;
    param_type p(5);
    assert(p.a() == 5);
    assert(p.b() == cuda::std::numeric_limits<long>::max());
  }
  {
    typedef cuda::std::uniform_int_distribution<long> D;
    typedef D::param_type param_type;
    param_type p(5, 10);
    assert(p.a() == 5);
    assert(p.b() == 10);
  }

  return 0;
}
