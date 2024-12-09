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

// transform_view::<iterator>::base

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    using TransformView = cuda::std::ranges::transform_view<MoveOnlyView, PlusOneMutable>;
    TransformView tv{};
    auto it  = tv.begin();
    using It = decltype(it);
    ASSERT_SAME_TYPE(decltype(static_cast<It&>(it).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<It&&>(it).base()), int*);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&>(it).base()), int* const&);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&&>(it).base()), int* const&);
    ASSERT_NOEXCEPT(it.base());
    assert(base(it.base()) == globalBuff);
    assert(base(cuda::std::move(it).base()) == globalBuff);
  }
  {
    using TransformView = cuda::std::ranges::transform_view<InputView, PlusOneMutable>;
    TransformView tv{};
    auto it  = tv.begin();
    using It = decltype(it);
    ASSERT_SAME_TYPE(decltype(static_cast<It&>(it).base()), const cpp20_input_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(static_cast<It&&>(it).base()), cpp20_input_iterator<int*>);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&>(it).base()), const cpp20_input_iterator<int*>&);
    ASSERT_SAME_TYPE(decltype(static_cast<const It&&>(it).base()), const cpp20_input_iterator<int*>&);
    ASSERT_NOEXCEPT(it.base());
    assert(base(it.base()) == globalBuff);
    assert(base(cuda::std::move(it).base()) == globalBuff);
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF

  return 0;
}
