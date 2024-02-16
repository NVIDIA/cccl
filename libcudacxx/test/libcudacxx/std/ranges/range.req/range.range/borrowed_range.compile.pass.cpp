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

// template<class T>
// concept borrowed_range;

#include <cuda/std/ranges>

#include "test_macros.h"

struct NotRange {
  TEST_HOST_DEVICE int begin() const;
  TEST_HOST_DEVICE int end() const;
};

struct Range {
  TEST_HOST_DEVICE int *begin();
  TEST_HOST_DEVICE int *end();
};

struct ConstRange {
  TEST_HOST_DEVICE int *begin() const;
  TEST_HOST_DEVICE int *end() const;
};

struct BorrowedRange {
  TEST_HOST_DEVICE int *begin() const;
  TEST_HOST_DEVICE int *end() const;
};

namespace cuda { namespace std { namespace ranges {
template<>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_borrowed_range<BorrowedRange> = true;
}}} // namespace cuda::std::ranges

static_assert(!cuda::std::ranges::borrowed_range<NotRange>);
static_assert(!cuda::std::ranges::borrowed_range<NotRange&>);
static_assert(!cuda::std::ranges::borrowed_range<const NotRange>);
static_assert(!cuda::std::ranges::borrowed_range<const NotRange&>);
static_assert(!cuda::std::ranges::borrowed_range<NotRange&&>);

static_assert(!cuda::std::ranges::borrowed_range<Range>);
static_assert( cuda::std::ranges::borrowed_range<Range&>);
static_assert(!cuda::std::ranges::borrowed_range<const Range>);
static_assert(!cuda::std::ranges::borrowed_range<const Range&>);
static_assert(!cuda::std::ranges::borrowed_range<Range&&>);

static_assert(!cuda::std::ranges::borrowed_range<ConstRange>);
static_assert( cuda::std::ranges::borrowed_range<ConstRange&>);
static_assert(!cuda::std::ranges::borrowed_range<const ConstRange>);
static_assert( cuda::std::ranges::borrowed_range<const ConstRange&>);
static_assert(!cuda::std::ranges::borrowed_range<ConstRange&&>);

static_assert( cuda::std::ranges::borrowed_range<BorrowedRange>);
static_assert( cuda::std::ranges::borrowed_range<BorrowedRange&>);
static_assert( cuda::std::ranges::borrowed_range<const BorrowedRange>);
static_assert( cuda::std::ranges::borrowed_range<const BorrowedRange&>);
static_assert( cuda::std::ranges::borrowed_range<BorrowedRange&&>);

int main(int, char**) {
  return 0;
}
