//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: LIBCUDACXX-has-no-incomplete-ranges

// Test the [[nodiscard]] extension in libc++.

// template<class I>
// unspecified iter_move;

#include <cuda/std/iterator>

struct WithADL
{
  WithADL() = default;
  __host__ __device__ constexpr decltype(auto) operator*() const noexcept;
  __host__ __device__ constexpr WithADL& operator++() noexcept;
  __host__ __device__ constexpr void operator++(int) noexcept;
  __host__ __device__ constexpr bool operator==(WithADL const&) const noexcept;
  __host__ __device__ friend constexpr auto iter_move(WithADL&)
  {
    return 0;
  }
};

int main(int, char**)
{
  int* noADL = nullptr;
  cuda::std::ranges::iter_move(noADL); // expected-warning {{ignoring return value of function declared with 'nodiscard'
                                       // attribute}}

  WithADL adl;
  cuda::std::ranges::iter_move(adl); // expected-warning {{ignoring return value of function declared with 'nodiscard'
                                     // attribute}}

  return 0;
}
