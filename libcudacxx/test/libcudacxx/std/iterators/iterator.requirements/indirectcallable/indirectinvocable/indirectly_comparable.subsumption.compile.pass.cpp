//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class I1, class I2, class R, class P1, class P2>
// concept indirectly_comparable;

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

template <class F>
  requires cuda::std::indirectly_comparable<int*, char*, F>
        && true // This true is an additional atomic constraint as a tie breaker
__host__ __device__ constexpr bool subsumes(F)
{
  return true;
}

template <class F>
  requires cuda::std::indirect_binary_predicate<F,
                                                cuda::std::projected<int*, cuda::std::identity>,
                                                cuda::std::projected<char*, cuda::std::identity>>
__host__ __device__ void subsumes(F);

template <class F>
  requires cuda::std::indirect_binary_predicate<F,
                                                cuda::std::projected<int*, cuda::std::identity>,
                                                cuda::std::projected<char*, cuda::std::identity>>
        && true // This true is an additional atomic constraint as a tie breaker
__host__ __device__ constexpr bool is_subsumed(F)
{
  return true;
}

template <class F>
  requires cuda::std::indirectly_comparable<int*, char*, F>
__host__ __device__ void is_subsumed(F);

static_assert(subsumes(cuda::std::less<int>()));
static_assert(is_subsumed(cuda::std::less<int>()));

int main(int, char**)
{
  return 0;
}
