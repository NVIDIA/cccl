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

// constexpr iterator& operator--() requires all-bidirectional<Const, Views...>;
// constexpr iterator operator--(int) requires all-bidirectional<Const, Views...>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class Iter>
concept canDecrement = requires(Iter it) { --it; } || requires(Iter it) { it--; };
#else
template <class Iter, class = void>
constexpr bool canDecrement = false;

template <class Iter>
constexpr bool canDecrement<Iter, cuda::std::void_t<decltype(--cuda::std::declval<Iter>())>> = true;
#endif // TEST_STD_VER <=2017

struct NonBidi : IntBufferView
{
#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
  template <cuda::std::size_t N>
  __host__ __device__ constexpr NonBidi(int (&b)[N])
      : IntBufferView(b)
  {}
#else
  using IntBufferView::IntBufferView;
#endif
  using iterator = forward_iterator<int*>;
  __host__ __device__ constexpr iterator begin() const
  {
    return iterator(buffer_);
  }
  __host__ __device__ constexpr sentinel_wrapper<iterator> end() const
  {
    return sentinel_wrapper<iterator>(iterator(buffer_ + size_));
  }
};

__host__ __device__ constexpr bool test()
{
  cuda::std::array<int, 4> a{1, 2, 3, 4};
  cuda::std::array<double, 3> b{4.1, 3.2, 4.3};
  {
    // all random access
    cuda::std::ranges::zip_view v(cuda::std::span<int>{a}, cuda::std::span<double>{b}, cuda::std::views::iota(0, 5));
    auto it    = v.end();
    using Iter = decltype(it);

    static_assert(cuda::std::is_same_v<decltype(--it), Iter&>);
    auto& it_ref = --it;
    assert(&it_ref == &it);

    assert(&(cuda::std::get<0>(*it)) == &(a[2]));
    assert(&(cuda::std::get<1>(*it)) == &(b[2]));
    assert(cuda::std::get<2>(*it) == 2);

    static_assert(cuda::std::is_same_v<decltype(it--), Iter>);
    it--;
    assert(&(cuda::std::get<0>(*it)) == &(a[1]));
    assert(&(cuda::std::get<1>(*it)) == &(b[1]));
    assert(cuda::std::get<2>(*it) == 1);
  }

  {
    // all bidi+
    int buffer[2] = {1, 2};

    cuda::std::ranges::zip_view v(BidiCommonView{buffer}, cuda::std::views::iota(0, 5));
    auto it    = v.begin();
    using Iter = decltype(it);

    ++it;
    ++it;

    static_assert(cuda::std::is_same_v<decltype(--it), Iter&>);
    auto& it_ref = --it;
    assert(&it_ref == &it);

    assert(it == ++v.begin());

    static_assert(cuda::std::is_same_v<decltype(it--), Iter>);
    auto tmp = it--;
    assert(it == v.begin());
    assert(tmp == ++v.begin());
  }

  {
    // non bidi
    int buffer[3] = {4, 5, 6};
    cuda::std::ranges::zip_view v(cuda::std::span<int>{a}, NonBidi{buffer});
    using Iter = cuda::std::ranges::iterator_t<decltype(v)>;
    static_assert(!canDecrement<Iter>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
