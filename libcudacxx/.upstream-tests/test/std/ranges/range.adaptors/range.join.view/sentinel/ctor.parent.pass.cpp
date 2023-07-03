//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr explicit sentinel(Parent& parent);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "../types.h"

__host__ __device__ constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  CopyableChild children[4] = {CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]), CopyableChild(buffer[3])};
  CopyableParent parent{children};
  cuda::std::ranges::join_view jv(parent);
  cuda::std::ranges::sentinel_t<decltype(jv)> sent(jv);
  assert(sent == cuda::std::ranges::next(jv.begin(), 16));

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)

  {
    // Test explicitness.
    using Parent = cuda::std::ranges::join_view<ParentView<ChildView>>;
    static_assert( cuda::std::is_constructible_v<cuda::std::ranges::sentinel_t<Parent>, Parent&>);
    static_assert(!cuda::std::is_convertible_v<cuda::std::ranges::sentinel_t<Parent>, Parent&>);
  }

  return 0;
}
