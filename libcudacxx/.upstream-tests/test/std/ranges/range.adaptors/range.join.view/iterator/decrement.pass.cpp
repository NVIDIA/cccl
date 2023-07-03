//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr iterator& operator--();
//              requires ref-is-glvalue && bidirectional_range<Base> &&
//                       bidirectional_range<range_reference_t<Base>> &&
//                       common_range<range_reference_t<Base>>;
// constexpr iterator operator--(int);
//              requires ref-is-glvalue && bidirectional_range<Base> &&
//                       bidirectional_range<range_reference_t<Base>> &&
//                       common_range<range_reference_t<Base>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "../types.h"

#if TEST_STD_VER > 17
template <class T>
concept CanPreDecrement = requires(T& t) { --t; };

template <class T>
concept CanPostDecrement = requires(T& t) { t--; };
#else
template <class T, class = void>
inline constexpr bool CanPreDecrement = false;

template <class T>
inline constexpr bool CanPreDecrement<T, cuda::std::void_t<decltype(--cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool CanPostDecrement = false;

template <class T>
inline constexpr bool CanPostDecrement<T, cuda::std::void_t<decltype(cuda::std::declval<T&>()--)>> = true;
#endif

template <class T>
__host__ __device__ constexpr void noDecrementTest(T&& jv) {
  auto iter = jv.begin();
  static_assert(!CanPreDecrement<decltype(iter)>);
  static_assert(!CanPostDecrement<decltype(iter)>);
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  {
    // outer == ranges::end
    cuda::std::ranges::join_view jv(buffer);
    auto iter = cuda::std::next(jv.begin(), 16);
    for (int i = 16; i != 0; --i) {
      assert(*--iter == i);
    }
  }

  {
    // outer == ranges::end
    cuda::std::ranges::join_view jv(buffer);
    auto iter = cuda::std::next(jv.begin(), 13);
    for (int i = 13; i != 0; --i) {
      assert(*--iter == i);
    }
  }

  {
    // outer != ranges::end
    cuda::std::ranges::join_view jv(buffer);
    auto iter = cuda::std::next(jv.begin(), 12);
    for (int i = 12; i != 0; --i) {
      assert(*--iter == i);
    }
  }

  {
    // outer != ranges::end
    cuda::std::ranges::join_view jv(buffer);
    auto iter = cuda::std::next(jv.begin());
    for (int i = 1; i != 0; --i) {
      assert(*--iter == i);
    }
  }

  {
    int small[2][1] = {{1}, {2}};
    cuda::std::ranges::join_view jv(small);
    auto iter = cuda::std::next(jv.begin(), 2);
    for (int i = 2; i != 0; --i) {
      assert(*--iter == i);
    }
  }

#if  TEST_STD_VER > 17 && TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
  if (!__builtin_is_constant_evaluated())
#endif
  {
    // skip empty inner
    BidiCommonInner inners[4] = {buffer[0], {nullptr, 0}, {nullptr, 0}, buffer[1]};
    cuda::std::ranges::join_view jv(inners);
    auto iter = jv.end();
    for (int i = 8; i != 0; --i) {
      assert(*--iter == i);
    }
  }

  {
    // basic type checking
    cuda::std::ranges::join_view jv(buffer);
    auto iter1 = cuda::std::ranges::next(jv.begin(), 4);
    using iterator = decltype(iter1);

    decltype(auto) iter2 = --iter1;
    static_assert(cuda::std::same_as<decltype(iter2), iterator&>);
    assert(&iter1 == &iter2);

    decltype(auto) iter3 = iter1--;
    static_assert(cuda::std::same_as<decltype(iter3), iterator>);
    assert(iter3 == cuda::std::next(iter1));
  }

  {
    // !ref-is-glvalue
    BidiCommonInner inners[2] = {buffer[0], buffer[1]};
    InnerRValue<BidiCommonOuter<BidiCommonInner>> outer{inners};
    cuda::std::ranges::join_view jv(outer);
    noDecrementTest(jv);
  }

  {
    // !bidirectional_range<Base>
    BidiCommonInner inners[2] = {buffer[0], buffer[1]};
    SimpleForwardCommonOuter<BidiCommonInner> outer{inners};
    cuda::std::ranges::join_view jv(outer);
    noDecrementTest(jv);
  }

  {
    // !bidirectional_range<range_reference_t<Base>>
    ForwardCommonInner inners[2] = {buffer[0], buffer[1]};
    cuda::std::ranges::join_view jv(inners);
    noDecrementTest(jv);
  }

  {
    // LWG3313 `join_view::iterator::operator--` is incorrectly constrained
    // `join_view::iterator` should not have `operator--` if
    // !common_range<range_reference_t<Base>>
    BidiNonCommonInner inners[2] = {buffer[0], buffer[1]};
    cuda::std::ranges::join_view jv(inners);
    auto iter = jv.begin();
    static_assert(!CanPreDecrement<decltype(iter)>);
    static_assert(!CanPostDecrement<decltype(iter)>);
    unused(iter);
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
