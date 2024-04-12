//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class T>
// concept range;

#include <cuda/std/ranges>

#include "test_range.h"

static_assert(cuda::std::ranges::range<test_range<cpp20_input_iterator>>);

struct incompatible_iterators
{
  __host__ __device__ int* begin();
  __host__ __device__ long* end();
};
static_assert(!cuda::std::ranges::range<incompatible_iterators>);

struct int_begin_int_end
{
  __host__ __device__ int begin();
  __host__ __device__ int end();
};
static_assert(!cuda::std::ranges::range<int_begin_int_end>);

struct iterator_begin_int_end
{
  __host__ __device__ int* begin();
  __host__ __device__ int end();
};
static_assert(!cuda::std::ranges::range<iterator_begin_int_end>);

struct int_begin_iterator_end
{
  __host__ __device__ int begin();
  __host__ __device__ int* end();
};
static_assert(!cuda::std::ranges::range<int_begin_iterator_end>);

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(!cuda::std::ranges::range<Holder<Incomplete>*>);
#endif

int main(int, char**)
{
  return 0;
}
