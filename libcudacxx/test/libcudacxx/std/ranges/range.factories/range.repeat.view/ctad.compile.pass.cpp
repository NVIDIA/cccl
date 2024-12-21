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

// template<class T, class Bound>
//    repeat_view(T, Bound) -> repeat_view<T, Bound>;

#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Empty
{};

int main(int, char**)
{
  Empty empty{};

  // clang-format off
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::repeat_view(Empty{})), cuda::std::ranges::repeat_view<Empty>>);
#if 0 // No passing in any compiler, maybe C++23 only fix?
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::repeat_view(empty)), cuda::std::ranges::repeat_view<Empty>>);
#endif //
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::repeat_view(cuda::std::move(empty))), cuda::std::ranges::repeat_view<Empty>>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::repeat_view(10, 1)), cuda::std::ranges::repeat_view<int, int>>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::repeat_view(10, 1U)), cuda::std::ranges::repeat_view<int, unsigned>>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::repeat_view(10, 1UL)), cuda::std::ranges::repeat_view<int, unsigned long>>);
  // clang-format on

  unused(empty);

  return 0;
}
