//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// stride() const noexcept;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Stride>
__host__ __device__ constexpr void test(Stride stride)
{
  {
    cuda::strided_iterator<int*, Stride> iter;
    assert(iter.stride() == ::cuda::std::__de_ice(Stride{}));
  }

  {
    cuda::strided_iterator<int*, Stride> iter{nullptr, stride};
    assert(iter.stride() == ::cuda::std::__de_ice(stride));
  }
}

__host__ __device__ constexpr bool test()
{
  test(0);
  test(Stride<2>{});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
