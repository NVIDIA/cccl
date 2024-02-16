//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::ranges::begin

#include <cuda/std/ranges>

struct NonBorrowedRange {
  TEST_HOST_DEVICE int* begin() const;
  TEST_HOST_DEVICE int* end() const;
};
static_assert(!cuda::std::ranges::enable_borrowed_range<NonBorrowedRange>);

// Verify that if the expression is an rvalue and `enable_borrowed_range` is false, `ranges::begin` is ill-formed.
TEST_HOST_DEVICE void test() {
  cuda::std::ranges::begin(NonBorrowedRange());
  // expected-error-re@-1 {{{{call to deleted function call operator in type 'const (cuda::std::ranges::)?__begin::__fn'}}}}
  // expected-error@-2  {{attempt to use a deleted function}}
}

int main(int, char**)
{
  return 0;
}
