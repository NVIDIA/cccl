//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Test that iota_view conforms to range and view concepts.

#include <cuda/std/ranges>

#include "types.h"

struct Decrementable
{
  using difference_type = int;

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const Decrementable&) const = default;
#else
  __host__ __device__ bool operator==(const Decrementable&) const;
  __host__ __device__ bool operator!=(const Decrementable&) const;

  __host__ __device__ bool operator<(const Decrementable&) const;
  __host__ __device__ bool operator<=(const Decrementable&) const;
  __host__ __device__ bool operator>(const Decrementable&) const;
  __host__ __device__ bool operator>=(const Decrementable&) const;
#endif

  __host__ __device__ Decrementable& operator++();
  __host__ __device__ Decrementable operator++(int);
  __host__ __device__ Decrementable& operator--();
  __host__ __device__ Decrementable operator--(int);
};

struct Incrementable
{
  using difference_type = int;

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const Incrementable&) const = default;
#else
  __host__ __device__ bool operator==(const Incrementable&) const;
  __host__ __device__ bool operator!=(const Incrementable&) const;

  __host__ __device__ bool operator<(const Incrementable&) const;
  __host__ __device__ bool operator<=(const Incrementable&) const;
  __host__ __device__ bool operator>(const Incrementable&) const;
  __host__ __device__ bool operator>=(const Incrementable&) const;
#endif

  __host__ __device__ Incrementable& operator++();
  __host__ __device__ Incrementable operator++(int);
};

static_assert(cuda::std::ranges::random_access_range<cuda::std::ranges::iota_view<int>>);
static_assert(cuda::std::ranges::random_access_range<const cuda::std::ranges::iota_view<int>>);
static_assert(cuda::std::ranges::bidirectional_range<cuda::std::ranges::iota_view<Decrementable>>);
static_assert(cuda::std::ranges::forward_range<cuda::std::ranges::iota_view<Incrementable>>);
static_assert(cuda::std::ranges::input_range<cuda::std::ranges::iota_view<NotIncrementable>>);
static_assert(cuda::std::ranges::view<cuda::std::ranges::iota_view<int>>);

int main(int, char**)
{
  return 0;
}
