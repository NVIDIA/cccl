//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// drop_view() requires default_initializable<V> = default;

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test() {
  cuda::std::ranges::drop_view<MoveOnlyView> dropView1;
  assert(cuda::std::ranges::begin(dropView1) == globalBuff);

  static_assert( cuda::std::is_default_constructible_v<cuda::std::ranges::drop_view<ForwardView>>);
  static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::drop_view<NoDefaultCtorForwardView>>);

  LIBCPP_STATIC_ASSERT( cuda::std::is_nothrow_default_constructible_v<cuda::std::ranges::drop_view<ForwardView>>);
  static_assert(!cuda::std::is_nothrow_default_constructible_v<ThrowingDefaultCtorForwardView>);
  static_assert(!cuda::std::is_nothrow_default_constructible_v<cuda::std::ranges::drop_view<ThrowingDefaultCtorForwardView>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
