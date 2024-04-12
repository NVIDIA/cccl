//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: nvcc-12.0 || nvcc-12.1 || nvcc-12.2 || nvcc-12.3
// nvbug 3885350

// template<class In, class Out>
// concept indirectly_copyable;

#include <cuda/std/iterator>

template <cuda::std::indirectly_readable I, class O>
__host__ __device__ constexpr bool indirectly_copyable_subsumption()
{
  return false;
}

template <class I, class O>
  requires cuda::std::indirectly_copyable<I, O>
__host__ __device__ constexpr bool indirectly_copyable_subsumption()
{
  return true;
}

static_assert(indirectly_copyable_subsumption<int*, int*>());

int main(int, char**)
{
  return 0;
}
