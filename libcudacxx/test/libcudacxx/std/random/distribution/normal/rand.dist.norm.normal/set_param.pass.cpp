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
// class normal_distribution;

// void param(const param_type& parm);

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    using D = cuda::std::normal_distribution<>;
    typedef D::param_type P;
    P p(0.25, 5.5);
    D d(0.75, 4);
    d.param(p);
    assert(d.param() == p);
  }

  return 0;
}
