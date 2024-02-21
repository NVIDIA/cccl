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

// transform_view::<iterator>::operator{++,--,+=,-=}

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::ranges::transform_view<MoveOnlyView, PlusOne> transformView{};
  auto iter = cuda::std::move(transformView).begin();
  assert((++iter).base() == globalBuff + 1);

  assert((iter++).base() == globalBuff + 1);
  assert(iter.base() == globalBuff + 2);

  assert((--iter).base() == globalBuff + 1);
  assert((iter--).base() == globalBuff + 1);
  assert(iter.base() == globalBuff);

  // Check that decltype(InputIter++) == void.
  ASSERT_SAME_TYPE(decltype(cuda::std::declval<
                            cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<InputView, PlusOne>>>()++),
                   void);

  assert((iter += 4).base() == globalBuff + 4);
  assert((iter -= 3).base() == globalBuff + 1);

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
