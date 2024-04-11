//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// cuda::std::forward_iterator;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "test_iterators.h"

static_assert(!cuda::std::forward_iterator<cpp17_input_iterator<int*>>);
static_assert(!cuda::std::forward_iterator<cpp20_input_iterator<int*>>);
static_assert(cuda::std::forward_iterator<forward_iterator<int*>>);
static_assert(cuda::std::forward_iterator<bidirectional_iterator<int*>>);
static_assert(cuda::std::forward_iterator<random_access_iterator<int*>>);
static_assert(cuda::std::forward_iterator<contiguous_iterator<int*>>);

static_assert(cuda::std::forward_iterator<int*>);
static_assert(cuda::std::forward_iterator<int const*>);
static_assert(cuda::std::forward_iterator<int volatile*>);
static_assert(cuda::std::forward_iterator<int const volatile*>);

struct not_input_iterator
{
  // using value_type = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::forward_iterator_tag;

  __host__ __device__ int operator*() const;

  __host__ __device__ not_input_iterator& operator++();
  __host__ __device__ not_input_iterator operator++(int);

#if TEST_STD_VER > 2017
  bool operator==(not_input_iterator const&) const = default;
#else
  __host__ __device__ bool operator==(const not_input_iterator&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const not_input_iterator&) const
  {
    return false;
  };
#endif
};
static_assert(cuda::std::input_or_output_iterator<not_input_iterator>);
static_assert(!cuda::std::input_iterator<not_input_iterator>);
static_assert(!cuda::std::forward_iterator<not_input_iterator>);

struct bad_iterator_tag
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  __host__ __device__ int operator*() const;

  __host__ __device__ bad_iterator_tag& operator++();
  __host__ __device__ bad_iterator_tag operator++(int);

#if TEST_STD_VER > 2017
  bool operator==(bad_iterator_tag const&) const = default;
#else
  __host__ __device__ bool operator==(const bad_iterator_tag&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const bad_iterator_tag&) const
  {
    return false;
  };
#endif
};
static_assert(!cuda::std::forward_iterator<bad_iterator_tag>);

struct not_incrementable
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::forward_iterator_tag;

  __host__ __device__ int operator*() const;

  __host__ __device__ not_incrementable& operator++();
  __host__ __device__ void operator++(int);

#if TEST_STD_VER > 2017
  bool operator==(not_incrementable const&) const = default;
#else
  __host__ __device__ bool operator==(const not_incrementable&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const not_incrementable&) const
  {
    return false;
  };
#endif
};
static_assert(!cuda::std::forward_iterator<not_incrementable>);

struct not_equality_comparable
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::forward_iterator_tag;

  __host__ __device__ int operator*() const;

  __host__ __device__ not_equality_comparable& operator++();
  __host__ __device__ not_equality_comparable operator++(int);

  __host__ __device__ bool operator==(not_equality_comparable const&) const = delete;
};
static_assert(!cuda::std::forward_iterator<not_equality_comparable>);

int main(int, char**)
{
  return 0;
}
