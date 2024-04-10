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

// template<sized_range R>
// using range_size_t = decltype(ranges::size(declval<R&>()));

#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/ranges>

#include "test_iterators.h"

#if TEST_STD_VER > 2017
template <class T>
concept has_range_size_t = requires { typename cuda::std::ranges::range_size_t<T>; };
#else
template <class T, class = void>
constexpr bool has_range_size_t = false;

template <class T>
constexpr bool has_range_size_t<T, cuda::std::void_t<cuda::std::ranges::range_size_t<T>>> = true;
#endif

struct A
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
  __host__ __device__ short size();
};
static_assert(cuda::std::same_as<cuda::std::ranges::range_size_t<A>, short>);
static_assert(cuda::std::same_as<cuda::std::ranges::range_size_t<A&>, short>);
static_assert(cuda::std::same_as<cuda::std::ranges::range_size_t<A&&>, short>);
static_assert(!has_range_size_t<const A>);
static_assert(!has_range_size_t<const A&>);
static_assert(!has_range_size_t<const A&&>);

struct B
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};
static_assert(cuda::std::same_as<cuda::std::ranges::range_size_t<B>, cuda::std::size_t>);
static_assert(cuda::std::same_as<cuda::std::ranges::range_size_t<B&>, cuda::std::size_t>);
static_assert(cuda::std::same_as<cuda::std::ranges::range_size_t<B&&>, cuda::std::size_t>);
static_assert(!has_range_size_t<const B>);
static_assert(!has_range_size_t<const B&>);
static_assert(!has_range_size_t<const B&&>);

struct C
{
  __host__ __device__ bidirectional_iterator<int*> begin();
  __host__ __device__ bidirectional_iterator<int*> end();
};
static_assert(!has_range_size_t<C>);

int main(int, char**)
{
  return 0;
}
