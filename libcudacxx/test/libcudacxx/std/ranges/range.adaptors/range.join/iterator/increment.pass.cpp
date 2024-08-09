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

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int)
//            requires ref-is-glvalue && forward_range<Base> &&
//                     forward_range<range_reference_t<Base>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // This way if we read past end we'll catch the error.
  int buffer1[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  int dummy         = 42;
  unused(dummy);
  int buffer2[2][4] = {{9, 10, 11, 12}, {13, 14, 15, 16}};

  // operator++(int);
  {
    cuda::std::ranges::join_view jv(buffer1);
    auto iter = jv.begin();
    for (int i = 1; i < 9; ++i)
    {
      assert(*iter++ == i);
    }
  }

  {
    using IntView       = ValueView<int>;
    IntView children[4] = {IntView(buffer1[0]), IntView(buffer1[1]), IntView(buffer2[0]), IntView(buffer2[1])};
    cuda::std::ranges::join_view jv(ValueView<IntView>{children});
    auto iter = jv.begin();
    for (int i = 1; i < 17; ++i)
    {
      assert(*iter == i);
      iter++;
    }

    ASSERT_SAME_TYPE(decltype(iter++), void);
  }

  {
    cuda::std::ranges::join_view jv(buffer1);
    auto iter = cuda::std::next(jv.begin(), 7);
    assert(*iter++ == 8);
    assert(iter == jv.end());
  }

  {
    int small[2][1] = {{1}, {2}};
    cuda::std::ranges::join_view jv(small);
    auto iter = jv.begin();
    for (int i = 1; i < 3; ++i)
    {
      assert(*iter++ == i);
    }
  }

  // Has some empty children.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer1[0], 4),
      CopyableChild(buffer1[1], 0),
      CopyableChild(buffer2[0], 1),
      CopyableChild(buffer2[1], 0)};
    auto jv   = cuda::std::ranges::join_view(ParentView(children));
    auto iter = jv.begin();
    assert(*iter == 1);
    iter++;
    assert(*iter == 2);
    iter++;
    assert(*iter == 3);
    iter++;
    assert(*iter == 4);
    iter++;
    assert(*iter == 9);
    iter++;
    assert(iter == jv.end());
  }

  // Parent is empty.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer1[0]), CopyableChild(buffer1[1]), CopyableChild(buffer2[0]), CopyableChild(buffer2[1])};
    cuda::std::ranges::join_view jv(ParentView(children, 0));
    assert(jv.begin() == jv.end());
  }

  // Parent size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer1[0])};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    auto iter = jv.begin();
    assert(*iter == 1);
    iter++;
    assert(*iter == 2);
    iter++;
    assert(*iter == 3);
    iter++;
    assert(*iter == 4);
    iter++;
    assert(iter == jv.end());
  }

  // Parent and child size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer1[0], 1)};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    auto iter = jv.begin();
    assert(*iter == 1);
    iter++;
    assert(iter == jv.end());
  }

  // Parent size is one child is empty
  {
    CopyableChild children[1] = {CopyableChild(buffer1[0], 0)};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    assert(jv.begin() == jv.end());
  }

  // Has all empty children.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer1[0], 0),
      CopyableChild(buffer1[1], 0),
      CopyableChild(buffer2[0], 0),
      CopyableChild(buffer2[1], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView(children));
    assert(jv.begin() == jv.end());
  }

  // First child is empty, others are not.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer1[0], 4),
      CopyableChild(buffer1[1], 0),
      CopyableChild(buffer2[0], 0),
      CopyableChild(buffer2[1], 0)};
    auto jv   = cuda::std::ranges::join_view(ParentView(children));
    auto iter = jv.begin();
    assert(*iter == 1);
    iter++;
    assert(*iter == 2);
    iter++;
    assert(*iter == 3);
    iter++;
    assert(*iter == 4);
    iter++;
    assert(iter == jv.end());
  }

  // Last child is empty, others are not.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer1[0], 4),
      CopyableChild(buffer1[1], 4),
      CopyableChild(buffer2[0], 4),
      CopyableChild(buffer2[1], 0)};
    auto jv   = cuda::std::ranges::join_view(ParentView(children));
    auto iter = jv.begin();
    for (int i = 1; i < 13; ++i)
    {
      assert(*iter == i);
      iter++;
    }
  }

  // operator++();
  {
    cuda::std::ranges::join_view jv(buffer1);
    auto iter = jv.begin();
    for (int i = 2; i < 9; ++i)
    {
      assert(*++iter == i);
    }
  }

  {
    using IntView       = ValueView<int>;
    IntView children[4] = {IntView(buffer1[0]), IntView(buffer1[1]), IntView(buffer2[0]), IntView(buffer2[1])};
    cuda::std::ranges::join_view jv(ValueView<IntView>{children});
    auto iter = jv.begin();
    for (int i = 2; i < 17; ++i)
    {
      assert(*++iter == i);
    }

    ASSERT_SAME_TYPE(decltype(++iter), decltype(iter)&);
  }

  {
    // check return value
    cuda::std::ranges::join_view jv(buffer1);
    auto iter      = jv.begin();
    using iterator = decltype(iter);

    decltype(auto) iter2 = ++iter;
    static_assert(cuda::std::is_same_v<decltype(iter2), iterator&>);
    assert(&iter2 == &iter);

    decltype(auto) iter3 = iter++;
    static_assert(cuda::std::same_as<decltype(iter3), iterator>);
    assert(cuda::std::next(iter3) == iter);
  }

  {
    // !ref-is-glvalue
    BidiCommonInner inners[2] = {buffer1[0], buffer1[1]};
    InnerRValue<BidiCommonOuter<BidiCommonInner>> outer{inners};
    cuda::std::ranges::join_view jv(outer);
    auto iter = jv.begin();
    static_assert(cuda::std::is_void_v<decltype(iter++)>);
    unused(iter);
  }

  {
    // !forward_range<Base>
    BufferView<int*> inners[2] = {buffer1[0], buffer1[1]};
    using Outer                = SimpleInputCommonOuter<BufferView<int*>>;
    cuda::std::ranges::join_view jv{Outer(inners)};
    auto iter = jv.begin();
    static_assert(cuda::std::is_void_v<decltype(iter++)>);
    unused(iter);
  }

  {
    // !forward_range<range_reference_t<Base>>
    InputCommonInner inners[1] = {buffer1[0]};
    cuda::std::ranges::join_view jv{inners};
    auto iter = jv.begin();
    static_assert(cuda::std::is_void_v<decltype(iter++)>);
    unused(iter);
  }

#ifdef _LIBCUDACXX_HAS_STRING
  {
    // Check stashing iterators (LWG3698: regex_iterator and join_view don't work together very well)
    cuda::std::ranges::join_view<StashingRange> jv;
    auto it = jv.begin();
    assert(*it == 'a');
    ++it;
    assert(*it == 'a');
    ++it;
    assert(*it == 'b');
    it++;
    assert(*it == 'a');
    it++;
    assert(*it == 'b');
    ++it;
    assert(*it == 'c');
  }
#endif // _LIBCUDACXX_HAS_STRING

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
