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

// constexpr auto begin() requires (!(simple-view<Views> && ...));
// constexpr auto begin() const requires (range<const Views> && ...);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

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
#else
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

struct NoConstBeginView : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin()
  {
    return nullptr;
  }
  __host__ __device__ int* end()
  {
    return nullptr;
  }
};

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // all underlying iterators should be at the begin position
    cuda::std::ranges::zip_view v(
      SizedRandomAccessView{buffer}, cuda::std::views::iota(0), cuda::std::ranges::single_view(2.));
    decltype(auto) val = *v.begin();
    static_assert(cuda::std::same_as<decltype(val), cuda::std::tuple<int&, int, double&>>);
    assert(val == cuda::std::make_tuple(1, 0, 2.0));
    assert(&(cuda::std::get<0>(val)) == &buffer[0]);
  }

  {
    // with empty range
    cuda::std::ranges::zip_view v(SizedRandomAccessView{buffer}, cuda::std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
  }

  {
    // underlying ranges all model simple-view
    cuda::std::ranges::zip_view v(SimpleCommon{buffer}, SimpleCommon{buffer});
    static_assert(cuda::std::is_same_v<decltype(v.begin()), decltype(cuda::std::as_const(v).begin())>);
    assert(v.begin() == cuda::std::as_const(v).begin());
    auto [x, y] = *cuda::std::as_const(v).begin();
    assert(&x == &buffer[0]);
    assert(&y == &buffer[0]);

    using View = decltype(v);
    static_assert(HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(!HasConstAndNonConstBegin<View>);
  }

  {
    // not all underlying ranges model simple-view
    cuda::std::ranges::zip_view v(SimpleCommon{buffer}, NonSimpleNonCommon{buffer});
    static_assert(!cuda::std::is_same_v<decltype(v.begin()), decltype(cuda::std::as_const(v).begin())>);
    assert(v.begin() == cuda::std::as_const(v).begin());
    auto [x, y] = *cuda::std::as_const(v).begin();
    assert(&x == &buffer[0]);
    assert(&y == &buffer[0]);

    using View = decltype(v);
    static_assert(!HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(HasConstAndNonConstBegin<View>);
  }

  {
    // underlying const R is not a range
    using View = cuda::std::ranges::zip_view<SimpleCommon, NoConstBeginView>;
    static_assert(!HasOnlyConstBegin<View>);
    static_assert(HasOnlyNonConstBegin<View>);
    static_assert(!HasConstAndNonConstBegin<View>);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
