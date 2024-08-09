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

// constexpr auto end() requires (!simple-view<V> && !common_range<V>)
// constexpr auto end() requires (!simple-view<V> && common_range<V>)
// constexpr auto end() const requires range<const V>
// constexpr auto end() const requires common_range<const V>

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

// | simple | common |      v.end()     | as_const(v)
// |        |        |                  |   .end()
// |--------|--------|------------------|---------------
// |   Y    |   Y    |  iterator<true>  | iterator<true>
// |   Y    |   N    |  sentinel<true>  | sentinel<true>
// |   N    |   Y    |  iterator<false> | iterator<true>
// |   N    |   N    |  sentinel<false> | sentinel<true>

// !range<const V>
#if TEST_STD_VER >= 2020
template <class T>
concept HasEnd = requires(T t) { t.end(); };

template <class T>
concept HasConstEnd = requires(const T ct) { ct.end(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasEnd = false;

template <class T>
inline constexpr bool HasEnd<T, cuda::std::void_t<decltype(cuda::std::declval<T>().end())>> = true;

template <class T, class = void>
inline constexpr bool HasConstEnd = false;

template <class T>
inline constexpr bool HasConstEnd<T, cuda::std::void_t<decltype(cuda::std::declval<const T>().end())>> = true;
#endif // TEST_STD_VER <= 2017
struct NoConstEndView : TupleBufferView
{
  DELEGATE_TUPLEBUFFERVIEW(NoConstEndView)
  __host__ __device__ constexpr cuda::std::tuple<int>* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr cuda::std::tuple<int>* end()
  {
    return buffer_ + size_;
  }
};

static_assert(HasEnd<cuda::std::ranges::elements_view<NoConstEndView, 0>>);
static_assert(!HasConstEnd<cuda::std::ranges::elements_view<NoConstEndView, 0>>);

__host__ __device__ constexpr bool test()
{
  cuda::std::tuple<int> buffer[] = {{1}, {2}, {3}};

  // simple-view && common_view
  {
    SimpleCommon v{buffer};
    auto ev = cuda::std::views::elements<0>(v);

    auto it           = ev.begin();
    decltype(auto) st = ev.end();
    assert(st == it + 3);

    auto const_it           = cuda::std::as_const(ev).begin();
    decltype(auto) const_st = cuda::std::as_const(ev).end();
    assert(const_st == const_it + 3);

    // Both iterator<true>
    static_assert(cuda::std::same_as<decltype(st), decltype(const_st)>);
    static_assert(cuda::std::same_as<decltype(st), decltype(it)>);
    static_assert(cuda::std::same_as<decltype(const_st), decltype(const_it)>);
  }

  // simple-view && !common_view
  {
    SimpleNonCommon v{buffer};
    auto ev = cuda::std::views::elements<0>(v);

    auto it           = ev.begin();
    decltype(auto) st = ev.end();
    assert(st == it + 3);

    auto const_it           = cuda::std::as_const(ev).begin();
    decltype(auto) const_st = cuda::std::as_const(ev).end();
    assert(const_st == const_it + 3);

    // Both iterator<true>
    static_assert(cuda::std::same_as<decltype(st), decltype(const_st)>);
    static_assert(!cuda::std::same_as<decltype(st), decltype(it)>);
    static_assert(!cuda::std::same_as<decltype(const_st), decltype(const_it)>);
  }

  // !simple-view && common_view
  {
    NonSimpleCommon v{buffer};
    auto ev = cuda::std::views::elements<0>(v);

    auto it           = ev.begin();
    decltype(auto) st = ev.end();
    assert(st == it + 3);

    auto const_it           = cuda::std::as_const(ev).begin();
    decltype(auto) const_st = cuda::std::as_const(ev).end();
    assert(const_st == const_it + 3);

    // iterator<false> and iterator<true>
    static_assert(!cuda::std::same_as<decltype(st), decltype(const_st)>);
    static_assert(cuda::std::same_as<decltype(st), decltype(it)>);
    static_assert(cuda::std::same_as<decltype(const_st), decltype(const_it)>);
  }

  // !simple-view && !common_view
  {
    NonSimpleNonCommon v{buffer};
    auto ev = cuda::std::views::elements<0>(v);

    auto it           = ev.begin();
    decltype(auto) st = ev.end();
    assert(st == it + 3);

    auto const_it           = cuda::std::as_const(ev).begin();
    decltype(auto) const_st = cuda::std::as_const(ev).end();
    assert(const_st == const_it + 3);

    // sentinel<false> and sentinel<true>
    static_assert(!cuda::std::same_as<decltype(st), decltype(const_st)>);
    static_assert(!cuda::std::same_as<decltype(st), decltype(it)>);
    static_assert(!cuda::std::same_as<decltype(const_st), decltype(const_it)>);
  }

  // LWG 3406 elements_view::begin() and elements_view::end() have incompatible constraints
  {
    cuda::std::tuple<int, int> x[] = {{0, 0}};
    cuda::std::ranges::subrange r  = {cuda::std::counted_iterator(x, 1), cuda::std::default_sentinel};
    auto v                         = r | cuda::std::views::elements<0>;
    assert(v.begin() != v.end());
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
