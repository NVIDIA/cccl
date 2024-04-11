//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class It, class T>
// concept output_iterator;

#include <cuda/std/cstddef>
#include <cuda/std/iterator>

#include "test_iterators.h"

struct T
{};
struct DerivedFromT : T
{};

static_assert(cuda::std::output_iterator<cpp17_output_iterator<int*>, int>);
static_assert(cuda::std::output_iterator<cpp17_output_iterator<int*>, short>);
static_assert(cuda::std::output_iterator<cpp17_output_iterator<int*>, long>);
static_assert(cuda::std::output_iterator<cpp17_output_iterator<T*>, T>);
static_assert(!cuda::std::output_iterator<cpp17_output_iterator<T const*>, T>);
static_assert(cuda::std::output_iterator<cpp17_output_iterator<T*>, T const>);
static_assert(cuda::std::output_iterator<cpp17_output_iterator<T*>, DerivedFromT>);
static_assert(!cuda::std::output_iterator<cpp17_output_iterator<DerivedFromT*>, T>);

static_assert(cuda::std::output_iterator<cpp20_output_iterator<int*>, int>);
static_assert(cuda::std::output_iterator<cpp20_output_iterator<int*>, short>);
static_assert(cuda::std::output_iterator<cpp20_output_iterator<int*>, long>);
static_assert(cuda::std::output_iterator<cpp20_output_iterator<T*>, T>);
static_assert(!cuda::std::output_iterator<cpp20_output_iterator<T const*>, T>);
static_assert(cuda::std::output_iterator<cpp20_output_iterator<T*>, T const>);
static_assert(cuda::std::output_iterator<cpp20_output_iterator<T*>, DerivedFromT>);
static_assert(!cuda::std::output_iterator<cpp20_output_iterator<DerivedFromT*>, T>);

// Not satisfied when the iterator is not an input_or_output_iterator
static_assert(!cuda::std::output_iterator<void, int>);
static_assert(!cuda::std::output_iterator<void (*)(), int>);
static_assert(!cuda::std::output_iterator<int&, int>);
static_assert(!cuda::std::output_iterator<T, int>);

// Not satisfied when we can't assign a T to the result of *it++
struct WrongPostIncrement
{
  using difference_type = cuda::std::ptrdiff_t;
  __host__ __device__ T const* operator++(int);
  __host__ __device__ WrongPostIncrement& operator++();
  __host__ __device__ T& operator*();
};
static_assert(cuda::std::input_or_output_iterator<WrongPostIncrement>);
static_assert(cuda::std::indirectly_writable<WrongPostIncrement, T>);
static_assert(!cuda::std::output_iterator<WrongPostIncrement, T>);

// Not satisfied when we can't assign a T to the result of *it (i.e. not indirectly_writable)
struct NotIndirectlyWritable
{
  using difference_type = cuda::std::ptrdiff_t;
  __host__ __device__ T* operator++(int);
  __host__ __device__ NotIndirectlyWritable& operator++();
  __host__ __device__ T const& operator*(); // const so we can't write to it
};
static_assert(cuda::std::input_or_output_iterator<NotIndirectlyWritable>);
static_assert(!cuda::std::indirectly_writable<NotIndirectlyWritable, T>);
static_assert(!cuda::std::output_iterator<NotIndirectlyWritable, T>);

int main(int, char**)
{
  return 0;
}
