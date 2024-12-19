//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr auto begin() requires (!simple-view<V>)
// { return ranges::begin(base_); }
//
// constexpr auto begin() const
//   requires range<const V> &&
//            indirect_unary_predicate<const Pred, iterator_t<const V>>
// { return ranges::begin(base_); }

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

// Test Constraints
#if TEST_STD_VER >= 2020
template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin = HasConstBegin<T> && requires(T& t, const T& ct) {
  requires !cuda::std::same_as<decltype(t.begin()), decltype(ct.begin())>;
};

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

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i > 5;
  }
};

static_assert(HasOnlyConstBegin<cuda::std::ranges::take_while_view<SimpleView, Pred>>);

static_assert(HasOnlyNonConstBegin<cuda::std::ranges::take_while_view<ConstNotRange, Pred>>);

static_assert(HasConstAndNonConstBegin<cuda::std::ranges::take_while_view<NonSimple, Pred>>);

struct NotPredForConst
{
  __host__ __device__ constexpr bool operator()(int& i) const
  {
    return i > 5;
  }
};
static_assert(HasOnlyNonConstBegin<cuda::std::ranges::take_while_view<NonSimple, NotPredForConst>>);

__host__ __device__ constexpr bool test()
{
  // simple-view
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    SimpleView v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.begin();
    static_assert(cuda::std::same_as<decltype(it1), int*>);
    assert(it1 == buffer);
    decltype(auto) it2 = cuda::std::as_const(twv).begin();
    static_assert(cuda::std::same_as<decltype(it2), int*>);
    assert(it2 == buffer);
  }

  // const not range
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    ConstNotRange v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.begin();
    static_assert(cuda::std::same_as<decltype(it1), int*>);
    assert(it1 == buffer);
  }

  // NonSimple
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.begin();
    static_assert(cuda::std::same_as<decltype(it1), int*>);
    assert(it1 == buffer);
    decltype(auto) it2 = cuda::std::as_const(twv).begin();
    static_assert(cuda::std::same_as<decltype(it2), const int*>);
    assert(it2 == buffer);
  }

  // NotPredForConst
  // LWG 3450: The const overloads of `take_while_view::begin/end` are underconstrained
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    cuda::std::ranges::take_while_view twv(v, NotPredForConst{});
    decltype(auto) it1 = twv.begin();
    static_assert(cuda::std::same_as<decltype(it1), int*>);
    assert(it1 == buffer);
  }

  return true;
}

int main(int, char**)
{
  test();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(test(), "");
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3
  return 0;
}
