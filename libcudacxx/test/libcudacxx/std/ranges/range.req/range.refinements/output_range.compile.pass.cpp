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

// template<class R, class T>
// concept output_range;

#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

struct T
{};

// Satisfied when it's a range and has the right iterator
struct GoodRange
{
  __host__ __device__ cpp17_output_iterator<T*> begin();
  __host__ __device__ sentinel end();
};
static_assert(cuda::std::ranges::range<GoodRange>);
static_assert(cuda::std::output_iterator<cuda::std::ranges::iterator_t<GoodRange>, T>);
static_assert(cuda::std::ranges::output_range<GoodRange, T>);

// Not satisfied when it's not a range
struct NotRange
{
  __host__ __device__ cpp17_output_iterator<T*> begin();
};
static_assert(!cuda::std::ranges::range<NotRange>);
static_assert(cuda::std::output_iterator<cuda::std::ranges::iterator_t<NotRange>, T>);
static_assert(!cuda::std::ranges::output_range<NotRange, T>);

// Not satisfied when the iterator is not an output_iterator
struct RangeWithBadIterator
{
  __host__ __device__ cpp17_input_iterator<T const*> begin();
  __host__ __device__ sentinel end();
};
static_assert(cuda::std::ranges::range<RangeWithBadIterator>);
static_assert(!cuda::std::output_iterator<cuda::std::ranges::iterator_t<RangeWithBadIterator>, T>);
static_assert(!cuda::std::ranges::output_range<RangeWithBadIterator, T>);

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>*, Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>*&, Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>*&&, Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const, Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const&, Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const&&, Holder<Incomplete>*>);

static_assert(cuda::std::ranges::output_range<Holder<Incomplete>* [10], Holder<Incomplete>*>);
static_assert(cuda::std::ranges::output_range<Holder<Incomplete>* (&) [10], Holder<Incomplete>*>);
static_assert(cuda::std::ranges::output_range<Holder<Incomplete>* (&&) [10], Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const[10], Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const (&)[10], Holder<Incomplete>*>);
static_assert(!cuda::std::ranges::output_range<Holder<Incomplete>* const (&&)[10], Holder<Incomplete>*>);
#endif

int main(int, char**)
{
  return 0;
}
