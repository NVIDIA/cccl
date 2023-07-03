//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class T>
//   single_view(T) -> single_view<T>;

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_iterators.h"

struct Empty {};

static_assert(cuda::std::same_as<
  decltype(cuda::std::ranges::single_view(Empty())),
  cuda::std::ranges::single_view<Empty>
>);

static_assert(cuda::std::same_as<
  decltype(cuda::std::ranges::single_view(cuda::std::declval<Empty&>())),
  cuda::std::ranges::single_view<Empty>
>);

static_assert(cuda::std::same_as<
  decltype(cuda::std::ranges::single_view(cuda::std::declval<Empty&&>())),
  cuda::std::ranges::single_view<Empty>
>);

int main(int, char**) {
  return 0;
}
