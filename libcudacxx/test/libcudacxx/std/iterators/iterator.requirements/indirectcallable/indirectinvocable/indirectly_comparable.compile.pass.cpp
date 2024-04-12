//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class I1, class I2, class R, class P1, class P2>
// concept indirectly_comparable;

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

struct Deref
{
  __host__ __device__ int operator()(int*) const;
};

static_assert(!cuda::std::indirectly_comparable<int, int, cuda::std::less<int>>); // not dereferenceable
static_assert(!cuda::std::indirectly_comparable<int*, int*, int>); // not a predicate
static_assert(cuda::std::indirectly_comparable<int*, int*, cuda::std::less<int>>);
static_assert(!cuda::std::indirectly_comparable<int**, int*, cuda::std::less<int>>);
static_assert(cuda::std::indirectly_comparable<int**, int*, cuda::std::less<int>, Deref>);
static_assert(!cuda::std::indirectly_comparable<int**, int*, cuda::std::less<int>, Deref, Deref>);
static_assert(!cuda::std::indirectly_comparable<int**, int*, cuda::std::less<int>, cuda::std::identity, Deref>);
static_assert(cuda::std::indirectly_comparable<int*, int**, cuda::std::less<int>, cuda::std::identity, Deref>);

int main(int, char**)
{
  return 0;
}
