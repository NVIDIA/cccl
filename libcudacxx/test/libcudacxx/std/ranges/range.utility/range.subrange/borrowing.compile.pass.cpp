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

// template<class I, class S, subrange_kind K>
//   inline constexpr bool enable_borrowed_range<subrange<I, S, K>> = true;

#include <cuda/std/ranges>

static_assert(cuda::std::ranges::borrowed_range<
              cuda::std::ranges::subrange<int*, const int*, cuda::std::ranges::subrange_kind::sized>>);
static_assert(
  cuda::std::ranges::borrowed_range<
    cuda::std::ranges::subrange<int*, cuda::std::unreachable_sentinel_t, cuda::std::ranges::subrange_kind::sized>>);
static_assert(
  cuda::std::ranges::borrowed_range<
    cuda::std::ranges::subrange<int*, cuda::std::unreachable_sentinel_t, cuda::std::ranges::subrange_kind::unsized>>);

int main(int, char**)
{
  return 0;
}
