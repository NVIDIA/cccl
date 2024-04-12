//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: msvc-19.16

// template<class T>
// concept input_iterator;

// cuda::std::ranges::forward_range

#include <cuda/std/iterator>
#include <cuda/std/ranges>

struct range
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};

template <cuda::std::ranges::range R>
  requires cuda::std::input_iterator<cuda::std::ranges::iterator_t<R>>
__host__ __device__ constexpr bool check_input_range_subsumption()
{
  return false;
}

template <cuda::std::ranges::input_range>
  requires true
__host__ __device__ constexpr bool check_input_range_subsumption()
{
  return true;
}

static_assert(check_input_range_subsumption<range>(), "");

template <cuda::std::ranges::input_range R>
  requires cuda::std::forward_iterator<cuda::std::ranges::iterator_t<R>>
__host__ __device__ constexpr bool check_forward_range_subsumption()
{
  return false;
}

template <cuda::std::ranges::forward_range>
  requires true
__host__ __device__ constexpr bool check_forward_range_subsumption()
{
  return true;
}

static_assert(check_forward_range_subsumption<range>(), "");

template <cuda::std::ranges::forward_range R>
  requires cuda::std::bidirectional_iterator<cuda::std::ranges::iterator_t<R>>
__host__ __device__ constexpr bool check_bidirectional_range_subsumption()
{
  return false;
}

template <cuda::std::ranges::bidirectional_range>
  requires true
__host__ __device__ constexpr bool check_bidirectional_range_subsumption()
{
  return true;
}

static_assert(check_bidirectional_range_subsumption<range>(), "");

template <cuda::std::ranges::bidirectional_range R>
  requires cuda::std::random_access_iterator<cuda::std::ranges::iterator_t<R>>
__host__ __device__ constexpr bool check_random_access_range_subsumption()
{
  return false;
}

template <cuda::std::ranges::random_access_range>
  requires true
__host__ __device__ constexpr bool check_random_access_range_subsumption()
{
  return true;
}

static_assert(check_random_access_range_subsumption<range>(), "");

template <cuda::std::ranges::random_access_range R>
  requires cuda::std::random_access_iterator<cuda::std::ranges::iterator_t<R>>
__host__ __device__ constexpr bool check_contiguous_range_subsumption()
{
  return false;
}

template <cuda::std::ranges::contiguous_range>
  requires true
__host__ __device__ constexpr bool check_contiguous_range_subsumption()
{
  return true;
}

static_assert(check_contiguous_range_subsumption<range>(), "");

int main(int, char**)
{
  return 0;
}
