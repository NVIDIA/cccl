//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class S, class I>
// concept sentinel_for;

#include <cuda/std/iterator>

#include "test_macros.h"

static_assert(cuda::std::sentinel_for<int*, int*>);
static_assert(!cuda::std::sentinel_for<int*, long*>);
struct nth_element_sentinel
{
  __host__ __device__ friend bool operator==(const nth_element_sentinel&, int*);
  __host__ __device__ friend bool operator==(int*, const nth_element_sentinel&);
  __host__ __device__ friend bool operator!=(const nth_element_sentinel&, int*);
  __host__ __device__ friend bool operator!=(int*, const nth_element_sentinel&);
};
static_assert(cuda::std::sentinel_for<nth_element_sentinel, int*>);

struct not_semiregular
{
  not_semiregular() = delete;
  __host__ __device__ friend bool operator==(const not_semiregular&, int*);
  __host__ __device__ friend bool operator==(int*, const not_semiregular&);
  __host__ __device__ friend bool operator!=(const not_semiregular&, int*);
  __host__ __device__ friend bool operator!=(int*, const not_semiregular&);
};
static_assert(!cuda::std::sentinel_for<not_semiregular, int*>);

struct weakly_equality_comparable_with_int
{
  __host__ __device__ friend bool operator==(const weakly_equality_comparable_with_int&, int);
  __host__ __device__ friend bool operator==(int, const weakly_equality_comparable_with_int&);
  __host__ __device__ friend bool operator!=(const weakly_equality_comparable_with_int&, int*);
  __host__ __device__ friend bool operator!=(int*, const weakly_equality_comparable_with_int&);
};
static_assert(!cuda::std::sentinel_for<weakly_equality_comparable_with_int, int>);

struct move_only_iterator
{
  using value_type      = int;
  using difference_type = cuda::std::ptrdiff_t;

  move_only_iterator() = default;

  move_only_iterator(move_only_iterator&&)            = default;
  move_only_iterator& operator=(move_only_iterator&&) = default;

  move_only_iterator(move_only_iterator const&)            = delete;
  move_only_iterator& operator=(move_only_iterator const&) = delete;

  __host__ __device__ value_type operator*() const;
  __host__ __device__ move_only_iterator& operator++();
  __host__ __device__ move_only_iterator operator++(int);

  __host__ __device__ bool operator==(move_only_iterator const&) const;
  __host__ __device__ bool operator!=(move_only_iterator const&) const;
};

#ifndef TEST_COMPILER_MSVC_2017
static_assert(cuda::std::movable<move_only_iterator> && !cuda::std::copyable<move_only_iterator>
              && cuda::std::input_or_output_iterator<move_only_iterator>
              && !cuda::std::sentinel_for<move_only_iterator, move_only_iterator>);
#endif // TEST_COMPILER_MSVC_2017

int main(int, char**)
{
  return 0;
}
