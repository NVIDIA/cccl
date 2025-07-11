//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iota_view(iterator first, see below last);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::iota_view commonView(SomeInt(0), SomeInt(10));
    cuda::std::ranges::iota_view<SomeInt, SomeInt> io(commonView.begin(), commonView.end());
    assert(cuda::std::ranges::next(io.begin(), 10) == io.end());
  }

  {
    cuda::std::ranges::iota_view unreachableSent(SomeInt(0));
    cuda::std::ranges::iota_view<SomeInt> io(unreachableSent.begin(), cuda::std::unreachable_sentinel);
    assert(cuda::std::ranges::next(io.begin(), 10) != io.end());
  }

  {
    cuda::std::ranges::iota_view differentTypes(SomeInt(0), IntComparableWith(SomeInt(10)));
    cuda::std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>> io(differentTypes.begin(), differentTypes.end());
    assert(cuda::std::ranges::next(io.begin(), 10) == io.end());
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
