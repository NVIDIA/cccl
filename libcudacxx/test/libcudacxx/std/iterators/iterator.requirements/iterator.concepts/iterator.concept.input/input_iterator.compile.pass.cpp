//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class T>
// concept input_iterator;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

static_assert(cuda::std::input_iterator<cpp17_input_iterator<int*>>);
static_assert(cuda::std::input_iterator<cpp20_input_iterator<int*>>);

struct no_explicit_iter_concept
{
  using value_type      = int;
  using difference_type = cuda::std::ptrdiff_t;

  no_explicit_iter_concept() = default;

  no_explicit_iter_concept(no_explicit_iter_concept&&)            = default;
  no_explicit_iter_concept& operator=(no_explicit_iter_concept&&) = default;

  no_explicit_iter_concept(no_explicit_iter_concept const&)            = delete;
  no_explicit_iter_concept& operator=(no_explicit_iter_concept const&) = delete;

  __host__ __device__ value_type operator*() const;

  __host__ __device__ no_explicit_iter_concept& operator++();
  __host__ __device__ void operator++(int);
};
#ifndef TEST_COMPILER_MSVC_2017
// ITER-CONCEPT is `random_access_iterator_tag` >:(
static_assert(cuda::std::input_iterator<no_explicit_iter_concept>);
#endif // TEST_COMPILER_MSVC_2017

static_assert(cuda::std::input_iterator<int*>);
static_assert(cuda::std::input_iterator<int const*>);
static_assert(cuda::std::input_iterator<int volatile*>);
static_assert(cuda::std::input_iterator<int const volatile*>);

struct not_weakly_incrementable
{
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  not_weakly_incrementable() = default;

  not_weakly_incrementable(not_weakly_incrementable&&)            = default;
  not_weakly_incrementable& operator=(not_weakly_incrementable&&) = default;

  not_weakly_incrementable(not_weakly_incrementable const&)            = delete;
  not_weakly_incrementable& operator=(not_weakly_incrementable const&) = delete;

  __host__ __device__ int operator*() const;

#if defined(TEST_COMPILER_MSVC) // nvbug4119179
  __host__ __device__ void operator++(int);
#else
  __host__ __device__ not_weakly_incrementable& operator++();
#endif // TEST_COMPILER_MSVC
};
static_assert(!cuda::std::input_or_output_iterator<not_weakly_incrementable>
              && !cuda::std::input_iterator<not_weakly_incrementable>);

struct not_indirectly_readable
{
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  not_indirectly_readable() = default;

  not_indirectly_readable(not_indirectly_readable&&)            = default;
  not_indirectly_readable& operator=(not_indirectly_readable&&) = default;

  not_indirectly_readable(not_indirectly_readable const&)            = delete;
  not_indirectly_readable& operator=(not_indirectly_readable const&) = delete;

  __host__ __device__ int operator*() const;

  __host__ __device__ not_indirectly_readable& operator++();
  __host__ __device__ void operator++(int);
};
static_assert(!cuda::std::indirectly_readable<not_indirectly_readable>
              && !cuda::std::input_iterator<not_indirectly_readable>);

struct bad_iterator_category
{
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using iterator_category = void;

  bad_iterator_category() = default;

  bad_iterator_category(bad_iterator_category&&)            = default;
  bad_iterator_category& operator=(bad_iterator_category&&) = default;

  bad_iterator_category(bad_iterator_category const&)            = delete;
  bad_iterator_category& operator=(bad_iterator_category const&) = delete;

  __host__ __device__ value_type operator*() const;

  __host__ __device__ bad_iterator_category& operator++();
  __host__ __device__ void operator++(int);
};
static_assert(!cuda::std::input_iterator<bad_iterator_category>);

struct bad_iterator_concept
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = void*;

  bad_iterator_concept() = default;

  bad_iterator_concept(bad_iterator_concept&&)            = default;
  bad_iterator_concept& operator=(bad_iterator_concept&&) = default;

  bad_iterator_concept(bad_iterator_concept const&)            = delete;
  bad_iterator_concept& operator=(bad_iterator_concept const&) = delete;

  __host__ __device__ value_type operator*() const;

  __host__ __device__ bad_iterator_concept& operator++();
  __host__ __device__ void operator++(int);
};
static_assert(!cuda::std::input_iterator<bad_iterator_concept>);

int main(int, char**)
{
  return 0;
}
