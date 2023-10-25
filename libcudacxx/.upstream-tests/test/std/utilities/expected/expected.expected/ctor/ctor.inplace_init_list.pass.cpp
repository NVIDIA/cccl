//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Older Clangs do not support the C++20 feature to constrain destructors

// template<class U, class... Args>
//   constexpr explicit expected(in_place_t, initializer_list<U> il, Args&&... args);
//
// Constraints: is_constructible_v<T, initializer_list<U>&, Args...> is true.
//
// Effects: Direct-non-list-initializes val with il, cuda::std::forward<Args>(args)....
//
// Postconditions: has_value() is true.
//
// Throws: Any exception thrown by the initialization of val.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#include <cuda/std/vector>
#endif

#include "MoveOnly.h"
#include "test_macros.h"

// Test Constraints:
#if defined(_LIBCUDACXX_HAS_VECTOR)
static_assert(
    cuda::std::is_constructible_v<cuda::std::expected<cuda::std::vector<int>, int>, cuda::std::in_place_t, cuda::std::initializer_list<int>>, "");
#endif

// !is_constructible_v<T, initializer_list<U>&, Args...>
static_assert(!cuda::std::is_constructible_v<cuda::std::expected<int, int>, cuda::std::in_place_t, cuda::std::initializer_list<int>>, "");

// test explicit
template <class T>
__host__ __device__ void conversion_test(T);

template <class T, class... Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  ImplicitlyConstructible_,
  requires(Args&&... args)(
    (conversion_test<T>({cuda::std::forward<Args>(args)...}))
  ));

template <class T, class... Args>
constexpr bool ImplicitlyConstructible = _LIBCUDACXX_FRAGMENT(ImplicitlyConstructible_, T, Args...);
static_assert(ImplicitlyConstructible<int, int>, "");

#if defined(_LIBCUDACXX_HAS_VECTOR)
static_assert(
    !ImplicitlyConstructible<cuda::std::expected<cuda::std::vector<int>, int>, cuda::std::in_place_t, cuda::std::initializer_list<int>>, "");
#endif

template <size_t N, class... Ts>
struct Data {
  int vec_[N] = {};
  cuda::std::tuple<Ts...> tuple_;

  _LIBCUDACXX_TEMPLATE(class... Us)
    _LIBCUDACXX_REQUIRES( cuda::std::is_constructible<cuda::std::tuple<Ts...>, Us&&...>::value)
  __host__ __device__ constexpr Data(cuda::std::initializer_list<int> il, Us&&... us) : tuple_(cuda::std::forward<Us>(us)...) {
      auto ibegin = il.begin();
      for (cuda::std::size_t i = 0; ibegin != il.end(); ++ibegin, ++i) {
        vec_[i] = *ibegin;
      }
    }
};

template<class Range1, class Range2>
__host__ __device__ constexpr bool equal(Range1&& lhs, Range2&& rhs) {
  auto* left = lhs + 0;
  auto* right = rhs.begin();

  for (; right != rhs.end(); ++left, ++right) {
    assert(*left == *right);
  }
  assert(right == rhs.end());
  return true;
}

__host__ __device__ constexpr bool test() {
  // no arg
  {
    cuda::std::expected<Data<3>, int> e(cuda::std::in_place, {1, 2, 3});
    assert(e.has_value());
    auto expectedList = {1, 2, 3};
    assert(equal(e.value().vec_, expectedList));
  }

  // one arg
  {
    cuda::std::expected<Data<3, MoveOnly>, int> e(cuda::std::in_place, {4, 5, 6}, MoveOnly(5));
    assert(e.has_value());
    auto expectedList = {4, 5, 6};
    assert((equal(e.value().vec_, expectedList)));
    assert(cuda::std::get<0>(e.value().tuple_) == 5);
  }

  // multi args
  {
    int i = 5;
    int j = 6;
    MoveOnly m(7);
    cuda::std::expected<Data<2, int&, int&&, MoveOnly>, int> e(cuda::std::in_place, {1, 2}, i, cuda::std::move(j), cuda::std::move(m));
    assert(e.has_value());
    auto expectedList = {1, 2};
    assert((equal(e.value().vec_, expectedList)));
    assert(&cuda::std::get<0>(e.value().tuple_) == &i);
    assert(&cuda::std::get<1>(e.value().tuple_) == &j);
    assert(cuda::std::get<2>(e.value().tuple_) == 7);
    assert(m.get() == 0);
  }

  return true;
}

__host__ __device__ void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Except {};

  struct Throwing {
    __host__ __device__ Throwing(cuda::std::initializer_list<int>, int) { throw Except{}; };
  };

  try {
    cuda::std::expected<Throwing, int> u(cuda::std::in_place, {1, 2}, 5);
    assert(false);
  } catch (Except) {
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test(), "");
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif // defined(_LIBCUDACXX_ADDRESSOF)
  testException();
  return 0;
}
