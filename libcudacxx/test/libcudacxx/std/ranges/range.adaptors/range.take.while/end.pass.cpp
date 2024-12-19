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

// constexpr auto end() requires (!simple-view<V>)
// { return sentinel<false>(ranges::end(base_), addressof(*pred_)); }
// constexpr auto end() const
//   requires range<const V> &&
//            indirect_unary_predicate<const Pred, iterator_t<const V>>
// { return sentinel<true>(ranges::end(base_), addressof(*pred_)); }

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

// Test Constraints
#if TEST_STD_VER >= 2020
template <class T>
concept HasConstEnd = requires(const T& ct) { ct.end(); };

template <class T>
concept HasEnd = requires(T& t) { t.end(); };

template <class T>
concept HasConstAndNonConstEnd = HasConstEnd<T> && requires(T& t, const T& ct) {
  requires !cuda::std::same_as<decltype(t.end()), decltype(ct.end())>;
};

template <class T>
concept HasOnlyNonConstEnd = HasEnd<T> && !HasConstEnd<T>;

template <class T>
concept HasOnlyConstEnd = HasConstEnd<T> && !HasConstAndNonConstEnd<T>;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv

template <class T, class = void>
inline constexpr bool HasConstEnd = false;

template <class T>
inline constexpr bool HasConstEnd<T, cuda::std::void_t<decltype(cuda::std::declval<const T&>().end())>> = true;

template <class T, class = void>
inline constexpr bool HasEnd = false;

template <class T>
inline constexpr bool HasEnd<T, cuda::std::void_t<decltype(cuda::std::declval<T&>().end())>> = true;

template <class T, class = void>
inline constexpr bool HasConstAndNonConstEnd = false;

template <class T>
inline constexpr bool HasConstAndNonConstEnd<
  T,
  cuda::std::void_t<cuda::std::enable_if_t<
    !cuda::std::same_as<decltype(cuda::std::declval<T&>().end()), decltype(cuda::std::declval<const T&>().end())>>>> =
  true;

template <class T>
inline constexpr bool HasOnlyNonConstEnd = HasEnd<T> && !HasConstEnd<T>;

template <class T>
inline constexpr bool HasOnlyConstEnd = HasConstEnd<T> && !HasConstAndNonConstEnd<T>;
#endif // TEST_STD_VER <= 2017

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i < 5;
  }
};

static_assert(HasOnlyConstEnd<cuda::std::ranges::take_while_view<SimpleView, Pred>>);

static_assert(HasOnlyNonConstEnd<cuda::std::ranges::take_while_view<ConstNotRange, Pred>>);

static_assert(HasConstAndNonConstEnd<cuda::std::ranges::take_while_view<NonSimple, Pred>>);

struct NotPredForConst
{
  __host__ __device__ constexpr bool operator()(int& i) const
  {
    return i > 5;
  }
};
static_assert(HasOnlyNonConstEnd<cuda::std::ranges::take_while_view<NonSimple, NotPredForConst>>);

__host__ __device__ constexpr bool test()
{
  // simple-view
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    SimpleView v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
    decltype(auto) it2 = cuda::std::as_const(twv).end();
    assert(it2 == buffer + 4);

#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
    static_assert(cuda::std::same_as<decltype(it1), decltype(it2)>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3
  }

  // const not range
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    ConstNotRange v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
  }

  // NonSimple
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    cuda::std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
    decltype(auto) it2 = cuda::std::as_const(twv).end();
    assert(it2 == buffer + 4);

    static_assert(!cuda::std::same_as<decltype(it1), decltype(it2)>);
  }

  // NotPredForConst
  // LWG 3450: The const overloads of `take_while_view::begin/end` are underconstrained
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    cuda::std::ranges::take_while_view twv(v, NotPredForConst{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
