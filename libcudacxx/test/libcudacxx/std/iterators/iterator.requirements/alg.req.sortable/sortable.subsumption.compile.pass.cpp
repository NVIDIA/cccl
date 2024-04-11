//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class I, class R = ranges::less, class P = identity>
//   concept sortable = see below;                            // since C++20

#include <cuda/std/functional>
#include <cuda/std/iterator>

template <class I, class R, class P>
__host__ __device__ void test_subsumption()
  requires cuda::std::permutable<I>;

template <class I, class R, class P>
__host__ __device__ void test_subsumption()
  requires cuda::std::indirect_strict_weak_order<R, cuda::std::projected<I, P>>;

template <class I, class R, class P>
__host__ __device__ constexpr bool test_subsumption()
  requires cuda::std::sortable<I, R, P>
{
  return true;
}

static_assert(test_subsumption<int*, cuda::std::ranges::less, cuda::std::identity>());

int main(int, char**)
{
  return 0;
}
