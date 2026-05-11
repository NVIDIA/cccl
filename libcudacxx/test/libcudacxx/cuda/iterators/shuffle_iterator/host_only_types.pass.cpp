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

struct host_bijection
{
  using index_type = uint32_t;

  host_bijection() {}
  host_bijection(const host_bijection&) {}
  host_bijection(host_bijection&&) {}
  host_bijection& operator=(const host_bijection&)
  {
    return *this;
  }
  host_bijection& operator=(host_bijection&&)
  {
    return *this;
  }
  ~host_bijection() {}

  template <class RNG>
  constexpr host_bijection(index_type, RNG&&) noexcept
  {}

  [[nodiscard]] constexpr index_type size() const noexcept
  {
    return 5;
  }

  [[nodiscard]] constexpr index_type operator()(index_type n) const noexcept
  {
    return __random_indices[n];
  }

  uint32_t __random_indices[5] = {4, 1, 2, 0, 3};
};

void test()
{
  // taken from host_bijection
  constexpr uint32_t random_indices[] = {4, 1, 2, 0, 3};
  cuda::shuffle_iterator iter{host_bijection{}};
  assert(iter[1] == random_indices[1]);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
