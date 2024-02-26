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

// constexpr explicit join_view(V base);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[4][4] = {{1111, 2222, 3333, 4444}, {555, 666, 777, 888}, {99, 1010, 1111, 1212}, {13, 14, 15, 16}};

  {
    ChildView children[4] = {ChildView(buffer[0]), ChildView(buffer[1]), ChildView(buffer[2]), ChildView(buffer[3])};
    auto jv               = cuda::std::ranges::join_view(ParentView{children});
    assert(cuda::std::move(jv).base().ptr_ == children);
  }

  {
    cuda::std::ranges::join_view jv(buffer);
    assert(jv.base().base() == buffer + 0);
  }

  {
    // Test explicitness.
    static_assert(
      cuda::std::is_constructible_v<cuda::std::ranges::join_view<ParentView<ChildView>>, ParentView<ChildView>>);
    static_assert(
      !cuda::std::is_convertible_v<cuda::std::ranges::join_view<ParentView<ChildView>>, ParentView<ChildView>>);
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
