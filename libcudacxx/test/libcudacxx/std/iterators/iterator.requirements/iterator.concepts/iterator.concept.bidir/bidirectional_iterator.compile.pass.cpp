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
// concept bidirectional_iterator;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "test_iterators.h"

static_assert(!cuda::std::bidirectional_iterator<cpp17_input_iterator<int*>>);
static_assert(!cuda::std::bidirectional_iterator<cpp20_input_iterator<int*>>);
static_assert(!cuda::std::bidirectional_iterator<forward_iterator<int*>>);
static_assert(cuda::std::bidirectional_iterator<bidirectional_iterator<int*>>);
static_assert(cuda::std::bidirectional_iterator<random_access_iterator<int*>>);
static_assert(cuda::std::bidirectional_iterator<contiguous_iterator<int*>>);

static_assert(cuda::std::bidirectional_iterator<int*>);
static_assert(cuda::std::bidirectional_iterator<int const*>);
static_assert(cuda::std::bidirectional_iterator<int volatile*>);
static_assert(cuda::std::bidirectional_iterator<int const volatile*>);

struct not_forward_iterator
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::bidirectional_iterator_tag;

  __host__ __device__ value_type operator*() const;

  __host__ __device__ not_forward_iterator& operator++();
  __host__ __device__ not_forward_iterator operator++(int);

  __host__ __device__ not_forward_iterator& operator--();
  __host__ __device__ not_forward_iterator& operator--(int);
};
static_assert(cuda::std::input_iterator<not_forward_iterator> && !cuda::std::forward_iterator<not_forward_iterator>
              && !cuda::std::bidirectional_iterator<not_forward_iterator>);

struct wrong_iterator_category
{
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using iterator_category = cuda::std::forward_iterator_tag;

  __host__ __device__ value_type& operator*() const;

  __host__ __device__ wrong_iterator_category& operator++();
  __host__ __device__ wrong_iterator_category operator++(int);

  __host__ __device__ wrong_iterator_category& operator--();
  __host__ __device__ wrong_iterator_category operator--(int);
#if TEST_STD_VER > 2017
  bool operator==(wrong_iterator_category const&) const = default;
#else
  __host__ __device__ bool operator==(const wrong_iterator_category&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const wrong_iterator_category&) const
  {
    return false;
  };
#endif
};
static_assert(!cuda::std::bidirectional_iterator<wrong_iterator_category>);

struct wrong_iterator_concept
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::forward_iterator_tag;

  __host__ __device__ value_type& operator*() const;

  __host__ __device__ wrong_iterator_concept& operator++();
  __host__ __device__ wrong_iterator_concept operator++(int);

  __host__ __device__ wrong_iterator_concept& operator--();
  __host__ __device__ wrong_iterator_concept operator--(int);

#if TEST_STD_VER > 2017
  bool operator==(wrong_iterator_concept const&) const = default;
#else
  __host__ __device__ bool operator==(const wrong_iterator_concept&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const wrong_iterator_concept&) const
  {
    return false;
  };
#endif
};
static_assert(!cuda::std::bidirectional_iterator<wrong_iterator_concept>);

struct no_predecrement
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::bidirectional_iterator_tag;

  __host__ __device__ value_type& operator*() const;

  __host__ __device__ no_predecrement& operator++();
  __host__ __device__ no_predecrement operator++(int);

  __host__ __device__ no_predecrement operator--(int);

#if TEST_STD_VER > 2017
  bool operator==(no_predecrement const&) const = default;
#else
  __host__ __device__ bool operator==(const no_predecrement&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const no_predecrement&) const
  {
    return false;
  };
#endif
};
static_assert(!cuda::std::bidirectional_iterator<no_predecrement>);

struct bad_predecrement
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::bidirectional_iterator_tag;

  __host__ __device__ value_type& operator*() const;

  __host__ __device__ bad_predecrement& operator++();
  __host__ __device__ bad_predecrement operator++(int);

  __host__ __device__ bad_predecrement operator--();
  __host__ __device__ bad_predecrement operator--(int);

#if TEST_STD_VER > 2017
  bool operator==(bad_predecrement const&) const = default;
#else
  __host__ __device__ bool operator==(const bad_predecrement&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const bad_predecrement&) const
  {
    return false;
  };
#endif
};
static_assert(!cuda::std::bidirectional_iterator<bad_predecrement>);

struct no_postdecrement
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::bidirectional_iterator_tag;

  __host__ __device__ value_type& operator*() const;

  __host__ __device__ no_postdecrement& operator++();
  __host__ __device__ no_postdecrement operator++(int);

  __host__ __device__ no_postdecrement& operator--();
#if defined(TEST_COMPILER_MSVC) //  single-argument function used for postfix "--" (anachronism)
  __host__ __device__ no_postdecrement& operator--(int) = delete;
#endif // TEST_COMPILER_MSVC

#if TEST_STD_VER > 2017
  bool operator==(no_postdecrement const&) const = default;
#else
  __host__ __device__ bool operator==(const no_postdecrement&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const no_postdecrement&) const
  {
    return false;
  };
#endif
};
static_assert(!cuda::std::bidirectional_iterator<no_postdecrement>);

struct bad_postdecrement
{
  using value_type       = int;
  using difference_type  = cuda::std::ptrdiff_t;
  using iterator_concept = cuda::std::bidirectional_iterator_tag;

  __host__ __device__ value_type& operator*() const;

  __host__ __device__ bad_postdecrement& operator++();
  __host__ __device__ bad_postdecrement operator++(int);

  __host__ __device__ bad_postdecrement& operator--();
  __host__ __device__ bad_postdecrement& operator--(int);

#if TEST_STD_VER > 2017
  bool operator==(bad_postdecrement const&) const = default;
#else
  __host__ __device__ bool operator==(const bad_postdecrement&) const
  {
    return true;
  };
  __host__ __device__ bool operator!=(const bad_postdecrement&) const
  {
    return false;
  };
#endif
};
static_assert(!cuda::std::bidirectional_iterator<bad_postdecrement>);

int main(int, char**)
{
  return 0;
}
