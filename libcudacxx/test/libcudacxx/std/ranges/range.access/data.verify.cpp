//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: nvrtc
// UNSUPPORTED: msvc-19.16

// std::ranges::data

#include <cuda/std/ranges>

struct NonBorrowedRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
static_assert(!cuda::std::ranges::enable_borrowed_range<NonBorrowedRange>);

// Verify that if the expression is an rvalue and `enable_borrowed_range` is false, `ranges::data` is ill-formed.
__host__ __device__ void test()
{
  cuda::std::ranges::data(NonBorrowedRange());
  // expected-error-re@-1 {{{{no matching function for call to object of type 'const (std::ranges::)?__data::__fn'}}}}
}

int main(int, char**)
{
  return 0;
}
