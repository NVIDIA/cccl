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

// template<class W, class Bound>
//   inline constexpr bool enable_borrowed_range<iota_view<W, Bound>> = true;

#include <cuda/std/ranges>

static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::iota_view<int, int>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::iota_view<int, cuda::std::unreachable_sentinel_t>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::iota_view<int*, int*>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::iota_view<int*, cuda::std::unreachable_sentinel_t>>);

int main(int, char**)
{
  return 0;
}
