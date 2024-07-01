//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class T>
//   inline constexpr bool enable_borrowed_range<common_view<T>> = enable_borrowed_range<T>;

#include <cuda/std/ranges>

// common_view can only wrap Ts that are `view<T> && !common_range<T>`, so we need to invent one.
struct Uncommon : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
  {
    return nullptr;
  };
  __host__ __device__ cuda::std::unreachable_sentinel_t end() const
  {
    return cuda::std::unreachable_sentinel;
  }
};

static_assert(!cuda::std::ranges::borrowed_range<cuda::std::ranges::common_view<Uncommon>>);
static_assert(
  !cuda::std::ranges::borrowed_range<cuda::std::ranges::common_view<cuda::std::ranges::owning_view<Uncommon>>>);
static_assert(cuda::std::ranges::borrowed_range<cuda::std::ranges::common_view<cuda::std::ranges::ref_view<Uncommon>>>);

int main(int, char**)
{
  return 0;
}
