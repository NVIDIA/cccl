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

#include "test_macros.h"

struct NotRange
{
  TEST_FUNC int begin() const;
  TEST_FUNC int end() const;
};

struct Range
{
  TEST_FUNC int* begin();
  TEST_FUNC int* end();
};

struct ConstRange
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

struct BorrowedRange
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool enable_borrowed_range<BorrowedRange> = true;
}

static_assert(!cuda::std::ranges::borrowed_range<NotRange>);
static_assert(!cuda::std::ranges::borrowed_range<NotRange&>);
static_assert(!cuda::std::ranges::borrowed_range<const NotRange>);
static_assert(!cuda::std::ranges::borrowed_range<const NotRange&>);
static_assert(!cuda::std::ranges::borrowed_range<NotRange&&>);

static_assert(!cuda::std::ranges::borrowed_range<Range>);
static_assert(cuda::std::ranges::borrowed_range<Range&>);
static_assert(!cuda::std::ranges::borrowed_range<const Range>);
static_assert(!cuda::std::ranges::borrowed_range<const Range&>);
static_assert(!cuda::std::ranges::borrowed_range<Range&&>);

static_assert(!cuda::std::ranges::borrowed_range<ConstRange>);
static_assert(cuda::std::ranges::borrowed_range<ConstRange&>);
static_assert(!cuda::std::ranges::borrowed_range<const ConstRange>);
static_assert(cuda::std::ranges::borrowed_range<const ConstRange&>);
static_assert(!cuda::std::ranges::borrowed_range<ConstRange&&>);

static_assert(cuda::std::ranges::borrowed_range<BorrowedRange>);
static_assert(cuda::std::ranges::borrowed_range<BorrowedRange&>);
static_assert(cuda::std::ranges::borrowed_range<const BorrowedRange>);
static_assert(cuda::std::ranges::borrowed_range<const BorrowedRange&>);
static_assert(cuda::std::ranges::borrowed_range<BorrowedRange&&>);

int main(int, char**)
{
  return 0;
}
