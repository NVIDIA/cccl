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

//  constexpr auto begin() requires (!simple-view<V>)
//  constexpr auto begin() const requires range<const V>

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

#if TEST_STD_VER >= 2020
template <class T>
concept HasConstBegin = requires(const T ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin =
  HasConstBegin<T> &&
  // because const begin and non-const begin returns different types (iterator<true>, iterator<false>)
  requires(T t, const T ct) { requires !cuda::std::same_as<decltype(t.begin()), decltype(ct.begin())>; };

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasConstBegin = false;

template <class T>
inline constexpr bool HasConstBegin<T, cuda::std::void_t<decltype(cuda::std::declval<const T&>().begin())>> = true;

template <class T, class = void>
inline constexpr bool HasBegin = false;

template <class T>
inline constexpr bool HasBegin<T, cuda::std::void_t<decltype(cuda::std::declval<T&>().begin())>> = true;

template <class T, class = void>
inline constexpr bool HasConstAndNonConstBegin = false;

template <class T>
inline constexpr bool HasConstAndNonConstBegin<
  T,
  cuda::std::void_t<cuda::std::enable_if_t<
    !cuda::std::same_as<decltype(cuda::std::declval<T&>().begin()), decltype(cuda::std::declval<const T&>().begin())>>>> =
  true;

template <class T>
inline constexpr bool HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
inline constexpr bool HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;
#endif // TEST_STD_VER <= 2017
struct NoConstBeginView : TupleBufferView
{
  DELEGATE_TUPLEBUFFERVIEW(NoConstBeginView)
  __host__ __device__ constexpr cuda::std::tuple<int>* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr cuda::std::tuple<int>* end()
  {
    return buffer_ + size_;
  }
};

// simple-view<V>
static_assert(HasOnlyConstBegin<cuda::std::ranges::elements_view<SimpleCommon, 0>>);

// !simple-view<V> && range<const V>
static_assert(HasConstAndNonConstBegin<cuda::std::ranges::elements_view<NonSimpleCommon, 0>>);

// !range<const V>
static_assert(HasOnlyNonConstBegin<cuda::std::ranges::elements_view<NoConstBeginView, 0>>);

__host__ __device__ constexpr bool test()
{
  cuda::std::tuple<int> buffer[] = {{1}, {2}};
  {
    // underlying iterator should be pointing to the first element
    auto ev   = cuda::std::views::elements<0>(buffer);
    auto iter = ev.begin();
    assert(&(*iter) == &cuda::std::get<0>(buffer[0]));
  }

  {
    // underlying range models simple-view
    auto v = cuda::std::views::elements<0>(SimpleCommon{buffer});
    static_assert(cuda::std::is_same_v<decltype(v.begin()), decltype(cuda::std::as_const(v).begin())>);
    assert(v.begin() == cuda::std::as_const(v).begin());
    auto&& r = *cuda::std::as_const(v).begin();
    assert(&r == &cuda::std::get<0>(buffer[0]));
  }

  {
    // underlying const R is not a range
    auto v   = cuda::std::views::elements<0>(NoConstBeginView{buffer});
    auto&& r = *v.begin();
    assert(&r == &cuda::std::get<0>(buffer[0]));
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
