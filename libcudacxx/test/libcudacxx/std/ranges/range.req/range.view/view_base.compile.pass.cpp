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

// <ranges>

// struct view_base { };

#include <cuda/std/ranges>
#include <cuda/std/type_traits>

static_assert(cuda::std::is_empty_v<cuda::std::ranges::view_base>);
static_assert(cuda::std::is_trivial_v<cuda::std::ranges::view_base>);

// Make sure we can inherit from it, as it's intended (that wouldn't be the
// case if e.g. it was marked as final).
struct View : cuda::std::ranges::view_base
{};

int main(int, char**)
{
  return 0;
}
