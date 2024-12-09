//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr decltype(auto) operator*() const;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  {
    cuda::std::ranges::join_view jv(buffer);
    auto iter = jv.begin();
    for (int i = 1; i < 17; ++i)
    {
      assert(*iter++ == i);
    }
  }
  {
    cuda::std::ranges::join_view jv(buffer);
    auto iter = cuda::std::next(jv.begin(), 15);
    assert(*iter++ == 16);
    assert(iter == jv.end());
  }
  {
    ChildView children[4] = {ChildView(buffer[0]), ChildView(buffer[1]), ChildView(buffer[2]), ChildView(buffer[3])};
    auto jv               = cuda::std::ranges::join_view(ParentView(children));
    auto iter             = jv.begin();
    for (int i = 1; i < 17; ++i)
    {
      assert(*iter == i);
      ++iter;
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
