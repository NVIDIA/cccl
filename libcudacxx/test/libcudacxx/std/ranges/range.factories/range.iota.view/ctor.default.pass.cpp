//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iota_view() requires default_initializable<W> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::iota_view<Int42<DefaultTo42>> io{};
    assert((*io.begin()).value_ == 42);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  static_assert(!cuda::std::default_initializable<Int42<ValueCtor>>);
  static_assert(cuda::std::default_initializable<Int42<DefaultTo42>>);

  return 0;
}
