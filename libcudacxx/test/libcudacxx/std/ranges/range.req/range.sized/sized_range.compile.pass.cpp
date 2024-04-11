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
// concept sized_range;

#include <cuda/std/ranges>

#include "test_iterators.h"

static_assert(cuda::std::ranges::sized_range<int[5]>);
static_assert(cuda::std::ranges::sized_range<int (&)[5]>);
static_assert(!cuda::std::ranges::sized_range<int (&)[]>);
static_assert(!cuda::std::ranges::sized_range<int[]>);

struct range_has_size
{
  __host__ __device__ bidirectional_iterator<int*> begin();
  __host__ __device__ bidirectional_iterator<int*> end();
  __host__ __device__ int size();
};
static_assert(cuda::std::ranges::sized_range<range_has_size>);
static_assert(!cuda::std::ranges::sized_range<range_has_size const>);

struct range_has_const_size
{
  __host__ __device__ bidirectional_iterator<int*> begin();
  __host__ __device__ bidirectional_iterator<int*> end();
  __host__ __device__ int size() const;
};
static_assert(cuda::std::ranges::sized_range<range_has_const_size>);
static_assert(!cuda::std::ranges::sized_range<range_has_const_size const>);

struct const_range_has_size
{
  __host__ __device__ bidirectional_iterator<int*> begin() const;
  __host__ __device__ bidirectional_iterator<int*> end() const;
  __host__ __device__ int size();
};
static_assert(cuda::std::ranges::sized_range<const_range_has_size>);
static_assert(cuda::std::ranges::range<const_range_has_size const>);
static_assert(!cuda::std::ranges::sized_range<const_range_has_size const>);

struct const_range_has_const_size
{
  __host__ __device__ bidirectional_iterator<int*> begin() const;
  __host__ __device__ bidirectional_iterator<int*> end() const;
  __host__ __device__ int size() const;
};
static_assert(cuda::std::ranges::sized_range<const_range_has_const_size>);
static_assert(cuda::std::ranges::sized_range<const_range_has_const_size const>);

struct sized_sentinel_range_has_size
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};
static_assert(cuda::std::ranges::sized_range<sized_sentinel_range_has_size>);
static_assert(!cuda::std::ranges::sized_range<sized_sentinel_range_has_size const>);

struct const_sized_sentinel_range_has_size
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
static_assert(cuda::std::ranges::sized_range<const_sized_sentinel_range_has_size>);
static_assert(cuda::std::ranges::sized_range<const_sized_sentinel_range_has_size const>);

struct non_range_has_size
{
  __host__ __device__ int size() const;
};
#if TEST_STD_VER > 2017
static_assert(requires(non_range_has_size const x) { unused(cuda::std::ranges::size(x)); });
#else
static_assert(cuda::std::invocable<decltype(cuda::std::ranges::size), non_range_has_size const>);
#endif
static_assert(!cuda::std::ranges::sized_range<non_range_has_size>);
static_assert(!cuda::std::ranges::sized_range<non_range_has_size const>);

int main(int, char**)
{
  return 0;
}
