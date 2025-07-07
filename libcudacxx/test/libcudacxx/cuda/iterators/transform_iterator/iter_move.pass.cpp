//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr decltype(auto) iter_move(const iterator& i)
//    noexcept(noexcept(invoke(i.parent_->fun_, *i.current_)))

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    cuda::transform_iterator iter{buffer, PlusOne{}};
    static_assert(!noexcept(cuda::std::ranges::iter_move(iter)));

    assert(cuda::std::ranges::iter_move(iter) == 1);
    assert(cuda::std::ranges::iter_move(iter + 2) == 3);

    static_assert(cuda::std::is_same_v<int, decltype(cuda::std::ranges::iter_move(iter))>);
    static_assert(cuda::std::is_same_v<int, decltype(cuda::std::ranges::iter_move(cuda::std::move(iter)))>);
  }

  {
    [[maybe_unused]] cuda::transform_iterator iter_noexcept{buffer, PlusOneNoexcept{}};
    static_assert(noexcept(cuda::std::ranges::iter_move(iter_noexcept)));

    [[maybe_unused]] cuda::transform_iterator iter_not_noexcept{buffer, PlusOneMutable{}};
    static_assert(!noexcept(cuda::std::ranges::iter_move(iter_not_noexcept)));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
