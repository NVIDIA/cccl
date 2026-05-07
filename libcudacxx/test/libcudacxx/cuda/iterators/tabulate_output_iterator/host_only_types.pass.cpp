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

#include "test_macros.h"

struct host_functor
{
  void operator()(const cuda::std::ptrdiff_t val, const int expected) const noexcept
  {
    assert(val == expected); // asserts that the assigned value matches the index
  }

  host_functor() {}
  host_functor(const host_functor&) {}
  host_functor(host_functor&&) {}
  host_functor& operator=(const host_functor&)
  {
    return *this;
  }
  host_functor& operator=(host_functor&&)
  {
    return *this;
  }
  ~host_functor() {}
};

void test()
{
  cuda::tabulate_output_iterator iter{host_functor{}, 10};
  iter[1] = 1 + 10;
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
