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

struct host_functor
{
  int operator()(const int val) const noexcept
  {
    return val + 42;
  }

  host_functor() {}
  host_functor(const host_functor&) {}
  host_functor(host_functor&&) {}
  host_functor& operator=(const host_functor&) {}
  host_functor& operator=(host_functor&&) {}
  ~host_functor() {}
};

void test()
{
  std::vector<int> vec{1, 2, 3, 4};
  cuda::transform_output_iterator iter{vec.begin(), host_functor{}};
  iter[1] = 1337;
  assert(vec[1] == 1337 + 42);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
