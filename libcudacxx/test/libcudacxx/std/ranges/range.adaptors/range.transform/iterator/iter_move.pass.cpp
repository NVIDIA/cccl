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

// friend constexpr decltype(auto) iter_move(const iterator& i)
//    noexcept(noexcept(invoke(i.parent_->fun_, *i.current_)))

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    cuda::std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOneMutable{});
    auto iter = transformView.begin();
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    ASSERT_NOT_NOEXCEPT(cuda::std::ranges::iter_move(iter));
#endif // !TEST_COMPILER_ICC

    assert(cuda::std::ranges::iter_move(iter) == 1);
    assert(cuda::std::ranges::iter_move(iter + 2) == 3);

    ASSERT_SAME_TYPE(int, decltype(cuda::std::ranges::iter_move(iter)));
    ASSERT_SAME_TYPE(int, decltype(cuda::std::ranges::iter_move(cuda::std::move(iter))));
  }

  {
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::ranges::iter_move(
      cuda::std::declval<
        cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<MoveOnlyView, PlusOneNoexcept>>&>()));
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    ASSERT_NOT_NOEXCEPT(cuda::std::ranges::iter_move(
      cuda::std::declval<
        cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>&>()));
#endif /// !TEST_COMPILER_ICC
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
