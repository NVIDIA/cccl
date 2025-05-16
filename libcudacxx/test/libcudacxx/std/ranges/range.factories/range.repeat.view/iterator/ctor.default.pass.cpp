//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iterator() = default;

#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::repeat_view<int>>;
  static_assert(cuda::std::is_default_constructible_v<Iter>);
  Iter iter;
  unused(iter);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
