//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class I1, class I2>
// concept indirectly_swappable;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

template <class I1, class I2>
  requires cuda::std::indirectly_readable<I1> && cuda::std::indirectly_readable<I2>
__host__ __device__ constexpr bool indirectly_swappable_subsumption()
{
  return false;
}

template <class I1, class I2>
  requires cuda::std::indirectly_swappable<I1, I2>
__host__ __device__ constexpr bool indirectly_swappable_subsumption()
{
  return true;
}

static_assert(indirectly_swappable_subsumption<int*, int*>());

int main(int, char**)
{
  return 0;
}
