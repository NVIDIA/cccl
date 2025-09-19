//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// constexpr common_type_t<duration> operator+() const;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  {
    const cuda::std::chrono::minutes m(3);
    cuda::std::chrono::minutes m2 = +m;
    assert(m.count() == m2.count());
  }

  {
    [[maybe_unused]] constexpr cuda::std::chrono::minutes m(3);
    [[maybe_unused]] constexpr cuda::std::chrono::minutes m2 = +m;
    static_assert(m.count() == m2.count(), "");
  }

  // P0548
  {
    using D10 = cuda::std::chrono::duration<int, cuda::std::ratio<10, 10>>;
    using D1  = cuda::std::chrono::duration<int, cuda::std::ratio<1, 1>>;
    [[maybe_unused]] D10 zero(0);
    [[maybe_unused]] D10 one(1);
    static_assert(cuda::std::is_same_v<decltype(+one), decltype(zero - one)>);
    static_assert(cuda::std::is_same_v<decltype(zero + one), D1>);
    static_assert(cuda::std::is_same_v<decltype(+one), D1>);
  }

  return 0;
}
