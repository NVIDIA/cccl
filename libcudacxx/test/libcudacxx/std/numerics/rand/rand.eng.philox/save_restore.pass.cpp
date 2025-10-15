
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

#include <cuda/std/__random/philox_engine.h>

#include "test_macros.h"

template <typename Engine>
void test()
{
  Engine e0;
  e0.discard(10000);
  std::stringstream ss;
  ss << e0;

  e0.discard(10000);
  Engine e1;
  ss >> e1;
  e1.discard(10000);
  assert(e0() == e1());
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 test<cuda::std::philox4x32>();
                 test<cuda::std::philox4x64>();
               }));
  return 0;
}
