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

// constexpr iterator<false> begin();
// constexpr iterator<true> begin() const
//   requires range<const V> &&
//            regular_invocable<const F&, range_reference_t<const V>>;

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

#if TEST_STD_VER >= 2020
template <class T>
concept BeginInvocable = requires(T t) { t.begin(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool BeginInvocable = false;

template <class T>
inline constexpr bool BeginInvocable<T, cuda::std::void_t<decltype(cuda::std::declval<T>().begin())>> = true;
#endif // TEST_STD_VER >= 2020

__host__ __device__ constexpr bool test()
{
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    cuda::std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOneMutable{});
    assert(transformView.begin().base() == buff);
    assert(*transformView.begin() == 1);
  }

  {
    cuda::std::ranges::transform_view transformView(ForwardView{buff}, PlusOneMutable{});
    assert(base(transformView.begin().base()) == buff);
    assert(*transformView.begin() == 1);
  }

  {
    cuda::std::ranges::transform_view transformView(InputView{buff}, PlusOneMutable{});
    assert(base(transformView.begin().base()) == buff);
    assert(*transformView.begin() == 1);
  }

  {
    const cuda::std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOne{});
    assert(*transformView.begin() == 1);
  }

  static_assert(!BeginInvocable<const cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>);

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
