//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// template<class T>
// concept borrowed_range;

#include <cuda/std/ranges>

struct NotRange
{
  __host__ __device__ int begin() const;
  __host__ __device__ int end() const;
};

struct Range
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};

struct ConstRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct BorrowedRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool enable_borrowed_range<BorrowedRange> = true;
}

static_assert(!cuda::std::ranges::borrowed_range<NotRange>, "");
static_assert(!cuda::std::ranges::borrowed_range<NotRange&>, "");
static_assert(!cuda::std::ranges::borrowed_range<const NotRange>, "");
static_assert(!cuda::std::ranges::borrowed_range<const NotRange&>, "");
static_assert(!cuda::std::ranges::borrowed_range<NotRange&&>, "");

static_assert(!cuda::std::ranges::borrowed_range<Range>, "");
static_assert(cuda::std::ranges::borrowed_range<Range&>, "");
static_assert(!cuda::std::ranges::borrowed_range<const Range>, "");
static_assert(!cuda::std::ranges::borrowed_range<const Range&>, "");
static_assert(!cuda::std::ranges::borrowed_range<Range&&>, "");

static_assert(!cuda::std::ranges::borrowed_range<ConstRange>, "");
static_assert(cuda::std::ranges::borrowed_range<ConstRange&>, "");
static_assert(!cuda::std::ranges::borrowed_range<const ConstRange>, "");
static_assert(cuda::std::ranges::borrowed_range<const ConstRange&>, "");
static_assert(!cuda::std::ranges::borrowed_range<ConstRange&&>, "");

static_assert(cuda::std::ranges::borrowed_range<BorrowedRange>, "");
static_assert(cuda::std::ranges::borrowed_range<BorrowedRange&>, "");
static_assert(cuda::std::ranges::borrowed_range<const BorrowedRange>, "");
static_assert(cuda::std::ranges::borrowed_range<const BorrowedRange&>, "");
static_assert(cuda::std::ranges::borrowed_range<BorrowedRange&&>, "");

int main(int, char**)
{
  return 0;
}
