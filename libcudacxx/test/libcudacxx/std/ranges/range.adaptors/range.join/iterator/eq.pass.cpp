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

// friend constexpr bool operator==(const iterator& x, const iterator& y);
//          requires ref-is-glvalue && equality_comparable<iterator_t<Base>> &&
//                   equality_comparable<iterator_t<range_reference_t<Base>>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  {
    cuda::std::ranges::join_view jv(buffer);
    auto iter1 = jv.begin();
    auto iter2 = jv.begin();
    assert(iter1 == iter2);
    iter1++;
    assert(iter1 != iter2);
    iter2++;
    assert(iter1 == iter2);

    assert(jv.begin() == cuda::std::as_const(jv).begin());
  }

  {
    // !ref-is-glvalue
    BidiCommonInner inners[2] = {buffer[0], buffer[1]};
    InnerRValue<BidiCommonOuter<BidiCommonInner>> outer{inners};
    cuda::std::ranges::join_view jv(outer);
    auto iter = jv.begin();
    static_assert(!cuda::std::equality_comparable<decltype(iter)>);
    unused(iter);
  }

  {
    // !forward_range<Base>
    using Inner = BufferView<int*>;
    using Outer = BufferView<cpp20_input_iterator<Inner*>, sentinel_wrapper<cpp20_input_iterator<Inner*>>>;
    static_assert(!cuda::std::equality_comparable<cuda::std::ranges::iterator_t<Outer>>);
    Inner inners[2] = {buffer[0], buffer[1]};
    cuda::std::ranges::join_view jv(Outer{inners});
    auto iter = jv.begin();
    static_assert(!cuda::std::equality_comparable<decltype(iter)>);
    unused(iter);
  }

  {
    // !equality_comparable<iterator_t<range_reference_t<Base>>>;
    using Inner     = BufferView<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>;
    Inner inners[1] = {buffer[0]};
    cuda::std::ranges::join_view jv{inners};
    auto iter = jv.begin();
    static_assert(!cuda::std::equality_comparable<decltype(iter)>);
    auto const_iter = cuda::std::as_const(jv).begin();
    static_assert(!cuda::std::equality_comparable<decltype(const_iter)>);
    unused(iter);
    unused(const_iter);
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
