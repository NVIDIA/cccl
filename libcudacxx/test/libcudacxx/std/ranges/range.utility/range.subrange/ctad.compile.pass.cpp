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

// class cuda::std::ranges::subrange;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

using FI = forward_iterator<int*>;
STATIC_TEST_GLOBAL_VAR FI fi{nullptr};
STATIC_TEST_GLOBAL_VAR int* ptr = nullptr;

static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(fi, fi)),
                                 cuda::std::ranges::subrange<FI, FI, cuda::std::ranges::subrange_kind::unsized>>);
static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(ptr, ptr, 0)),
                                 cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>>);
static_assert(
  cuda::std::same_as<decltype(cuda::std::ranges::subrange(ptr, nullptr, 0)),
                     cuda::std::ranges::subrange<int*, cuda::std::nullptr_t, cuda::std::ranges::subrange_kind::sized>>);

struct ForwardRange
{
  __host__ __device__ forward_iterator<int*> begin() const;
  __host__ __device__ forward_iterator<int*> end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<ForwardRange> = true;

struct SizedRange
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<SizedRange> = true;

static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(ForwardRange())),
                                 cuda::std::ranges::subrange<FI, FI, cuda::std::ranges::subrange_kind::unsized>>);
static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(SizedRange())),
                                 cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>>);
static_assert(cuda::std::same_as<decltype(cuda::std::ranges::subrange(SizedRange(), 8)),
                                 cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>>);

int main(int, char**)
{
  unused(fi);
  unused(ptr);

  return 0;
}
