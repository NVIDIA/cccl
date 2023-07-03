//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr iterator(Parent& parent, OuterIter outer);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

using NonDefaultCtrIter = cpp20_input_iterator<int*>;
static_assert(!cuda::std::default_initializable<NonDefaultCtrIter>);

using NonDefaultCtrIterView = BufferView<NonDefaultCtrIter, sentinel_wrapper<NonDefaultCtrIter>>;
static_assert(cuda::std::ranges::input_range<NonDefaultCtrIterView>);

__host__ __device__ constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  {
    CopyableChild children[4] = {CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]),
                                 CopyableChild(buffer[3])};
    CopyableParent parent{children};
    cuda::std::ranges::join_view jv(parent);
    cuda::std::ranges::iterator_t<decltype(jv)> iter(jv, cuda::std::ranges::begin(parent));
    assert(*iter == 1);
  }

  {
    // LWG3569 Inner iterator not default_initializable
    // With the current spec, the constructor under test invokes Inner iterator's default constructor
    // even if it is not default constructible
    // This test is checking that this constructor can be invoked with an inner range with non default
    // constructible iterator
    NonDefaultCtrIterView inners[] = {buffer[0], buffer[1]};
    auto outer = cuda::std::views::all(inners);
    cuda::std::ranges::join_view jv(outer);
    cuda::std::ranges::iterator_t<decltype(jv)> iter(jv, cuda::std::ranges::begin(outer));
    assert(*iter == 1);
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
