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

// cuda::std::ranges::end

#include <cuda/std/ranges>

struct NonBorrowedRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
static_assert(!cuda::std::ranges::enable_borrowed_range<NonBorrowedRange>);

// Verify that if the expression is an rvalue and `enable_borrowed_range` is false, `ranges::end` is ill-formed.
__host__ __device__ void test()
{
  cuda::std::ranges::end(NonBorrowedRange());
  // expected-error-re@-1 {{{{call to deleted function call operator in type 'const
  // (cuda::std::ranges::)?__end::__fn'}}}} expected-error@-2  {{attempt to use a deleted function}}
}

int main(int, char**)
{
  return 0;
}
