//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class In, class Out>
// concept indirectly_copyable_storable;

#include <cuda/std/iterator>

template <class I, class O>
  requires cuda::std::indirectly_copyable<I, O>
__host__ __device__ constexpr bool indirectly_copyable_storable_subsumption()
{
  return false;
}

template <class I, class O>
  requires cuda::std::indirectly_copyable_storable<I, O>
__host__ __device__ constexpr bool indirectly_copyable_storable_subsumption()
{
  return true;
}

#ifndef __NVCOMPILER // nvbug 3885350
static_assert(indirectly_copyable_storable_subsumption<int*, int*>());
#endif

int main(int, char**)
{
  return 0;
}
