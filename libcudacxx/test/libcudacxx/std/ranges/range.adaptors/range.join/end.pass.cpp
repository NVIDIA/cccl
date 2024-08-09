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

// constexpr auto end();
// constexpr auto end() const;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "types.h"

#if TEST_STD_VER >= 2020
template <class T>
concept HasConstEnd = requires(const T& t) { t.end(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasConstEnd = false;

template <class T>
inline constexpr bool HasConstEnd<T, cuda::std::void_t<decltype(cuda::std::declval<const T&>().end())>> = true;
#endif // TEST_STD_VER <= 2017

// | ID | outer  | outer   | outer  | inner | inner   | inner  |     end()     |    end()     |
// |    | simple | forward | common | l_ref | forward | common |               |    const     |
// |----|--------|---------|--------|-------|---------|--------|---------------|--------------|
// | 1  |   Y    |   Y     |   Y    |   Y   |    Y    |   Y    |iterator<true> |iterator<true>|
// | 2  |   Y    |   Y     |   Y    |   Y   |    Y    |   N    |sentinel<true> |sentinel<true>|
// | 3  |   Y    |   Y     |   Y    |   Y   |    N    |   Y    |sentinel<true> |sentinel<true>|
// | 4  |   Y    |   Y     |   Y    |   N   |    Y    |   Y    |sentinel<true> |      -       |
// | 5  |   Y    |   Y     |   N    |   Y   |    Y    |   Y    |sentinel<true> |sentinel<true>|
// | 6  |   Y    |   N     |   Y    |   Y   |    Y    |   Y    |sentinel<true> |      -       |
// | 7  |   N    |   Y     |   Y    |   Y   |    Y    |   Y    |iterator<false>|iterator<true>|
// | 8  |   N    |   Y     |   Y    |   Y   |    Y    |   N    |sentinel<false>|sentinel<true>|
// | 9  |   N    |   Y     |   Y    |   Y   |    N    |   Y    |sentinel<false>|sentinel<true>|
// | 10 |   N    |   Y     |   Y    |   N   |    Y    |   Y    |sentinel<false>|      -       |
// | 11 |   N    |   Y     |   N    |   Y   |    Y    |   Y    |sentinel<false>|sentinel<true>|
// | 12 |   N    |   N     |   Y    |   Y   |    Y    |   Y    |sentinel<false>|      -       |
//
//

struct ConstNotRange : cuda::std::ranges::view_base
{
  __host__ __device__ const ChildView* begin();
  __host__ __device__ const ChildView* end();
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int buffer[4][4] = {{1111, 2222, 3333, 4444}, {555, 666, 777, 888}, {99, 1010, 1111, 1212}, {13, 14, 15, 16}};

  {
    // test ID 1
    ForwardCommonInner inners[4] = {buffer[0], buffer[1], buffer[2], buffer[3]};
    SimpleForwardCommonOuter<ForwardCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 16));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 16));

    static_assert(HasConstEnd<decltype(jv)>);
    static_assert(cuda::std::same_as<decltype(jv.end()), decltype(cuda::std::as_const(jv).end())>);
    static_assert(cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 2
    ForwardNonCommonInner inners[3] = {buffer[0], buffer[1], buffer[2]};
    SimpleForwardCommonOuter<ForwardNonCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 12));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 12));

    static_assert(HasConstEnd<decltype(jv)>);
    static_assert(cuda::std::same_as<decltype(jv.end()), decltype(cuda::std::as_const(jv).end())>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 3
    InputCommonInner inners[3] = {buffer[0], buffer[1], buffer[2]};
    SimpleForwardCommonOuter<InputCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 12));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 12));

    static_assert(HasConstEnd<decltype(jv)>);
    static_assert(cuda::std::same_as<decltype(jv.end()), decltype(cuda::std::as_const(jv).end())>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 4
    ForwardCommonInner inners[2] = {buffer[0], buffer[1]};
    InnerRValue<SimpleForwardCommonOuter<ForwardCommonInner>> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 8));

    static_assert(!HasConstEnd<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 5
    ForwardCommonInner inners[4] = {buffer[0], buffer[1], buffer[2], buffer[3]};
    SimpleForwardNonCommonOuter<ForwardCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 16));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 16));

    static_assert(HasConstEnd<decltype(jv)>);
    static_assert(cuda::std::same_as<decltype(jv.end()), decltype(cuda::std::as_const(jv).end())>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 6
    ForwardCommonInner inners[4] = {buffer[0], buffer[1], buffer[2], buffer[3]};
    SimpleInputCommonOuter<ForwardCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 16));

    static_assert(!HasConstEnd<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 7
    ForwardCommonInner inners[1] = {buffer[0]};
    NonSimpleForwardCommonOuter<ForwardCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 4));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 4));

    static_assert(HasConstEnd<decltype(jv)>);
    static_assert(!cuda::std::same_as<decltype(jv.end()), decltype(cuda::std::as_const(jv).end())>);
    static_assert(cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 8
    ForwardNonCommonInner inners[3] = {buffer[0], buffer[1], buffer[2]};
    NonSimpleForwardCommonOuter<ForwardNonCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 12));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 12));

    static_assert(HasConstEnd<decltype(jv)>);
    static_assert(!cuda::std::same_as<decltype(jv.end()), decltype(cuda::std::as_const(jv).end())>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 9
    InputCommonInner inners[3] = {buffer[0], buffer[1], buffer[2]};
    NonSimpleForwardCommonOuter<InputCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 12));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 12));

    static_assert(HasConstEnd<decltype(jv)>);
    static_assert(!cuda::std::same_as<decltype(jv.end()), decltype(cuda::std::as_const(jv).end())>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 10
    ForwardCommonInner inners[2] = {buffer[0], buffer[1]};
    InnerRValue<NonSimpleForwardCommonOuter<ForwardCommonInner>> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 8));

    static_assert(!HasConstEnd<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 11
    ForwardCommonInner inners[4] = {buffer[0], buffer[1], buffer[2], buffer[3]};
    NonSimpleForwardNonCommonOuter<ForwardCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 16));
    assert(cuda::std::as_const(jv).end() == cuda::std::ranges::next(cuda::std::as_const(jv).begin(), 16));

    static_assert(HasConstEnd<decltype(jv)>);
    static_assert(!cuda::std::same_as<decltype(jv.end()), decltype(cuda::std::as_const(jv).end())>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    // test ID 12
    ForwardCommonInner inners[4] = {buffer[0], buffer[1], buffer[2], buffer[3]};
    NonSimpleInputCommonOuter<ForwardCommonInner> outer{inners};

    cuda::std::ranges::join_view jv(outer);
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 16));

    static_assert(!HasConstEnd<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<decltype(jv)>);
    static_assert(!cuda::std::ranges::common_range<const decltype(jv)>);
  }

  {
    cuda::std::ranges::join_view jv(ConstNotRange{});
    static_assert(!HasConstEnd<decltype(jv)>);
    unused(jv);
  }

  // Has some empty children.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0], 4),
      CopyableChild(buffer[1], 0),
      CopyableChild(buffer[2], 1),
      CopyableChild(buffer[3], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView(children));
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 5));
  }

  // Parent is empty.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0]), CopyableChild(buffer[1]), CopyableChild(buffer[2]), CopyableChild(buffer[3])};
    cuda::std::ranges::join_view jv(ParentView(children, 0));
    assert(jv.end() == jv.begin());
  }

  // Parent size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer[0])};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 4));
  }

  // Parent and child size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer[0], 1)};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    assert(jv.end() == cuda::std::ranges::next(jv.begin()));
  }

  // Parent size is one child is empty
  {
    CopyableChild children[1] = {CopyableChild(buffer[0], 0)};
    cuda::std::ranges::join_view jv(ParentView(children, 1));
    assert(jv.end() == jv.begin());
  }

  // Has all empty children.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0], 0),
      CopyableChild(buffer[1], 0),
      CopyableChild(buffer[2], 0),
      CopyableChild(buffer[3], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView(children));
    assert(jv.end() == jv.begin());
  }

  // First child is empty, others are not.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0], 4),
      CopyableChild(buffer[1], 0),
      CopyableChild(buffer[2], 0),
      CopyableChild(buffer[3], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView(children));
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 4));
  }

  // Last child is empty, others are not.
  {
    CopyableChild children[4] = {
      CopyableChild(buffer[0], 4),
      CopyableChild(buffer[1], 4),
      CopyableChild(buffer[2], 4),
      CopyableChild(buffer[3], 0)};
    auto jv = cuda::std::ranges::join_view(ParentView(children));
    assert(jv.end() == cuda::std::ranges::next(jv.begin(), 12));
  }

#ifdef _LIBCUDACXX_HAS_STRING
  // LWG3700: The `const begin` of the `join_view` family does not require `InnerRng` to be a range
  {
    cuda::std::ranges::join_view<ConstNonJoinableRange> jv;
    static_assert(!HasConstEnd<decltype(jv)>);
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
