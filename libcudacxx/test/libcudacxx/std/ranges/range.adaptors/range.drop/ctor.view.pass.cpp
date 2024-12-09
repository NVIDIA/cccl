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

// constexpr drop_view(V base, range_difference_t<V> count);

#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  cuda::std::ranges::drop_view dropView1(MoveOnlyView(), 4);
  assert(dropView1.size() == 4);
  assert(dropView1.begin() == globalBuff + 4);

  cuda::std::ranges::drop_view dropView2(ForwardView(), 4);
  assert(base(dropView2.begin()) == globalBuff + 4);

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
