//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile

// UNSUPPORTED: nvrtc

#include <cuda/iterator>
#include <cuda/std/cassert>

#include <vector>

#include "test_macros.h"

void test()
{
  std::vector<int> vec{1, 2, 3, 4};
  cuda::zip_iterator iter{vec.begin(), vec.begin() + 1};
  assert(cuda::std::get<0>(iter[1]) == vec[1]);
  assert(cuda::std::get<1>(iter[1]) == vec[2]);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
