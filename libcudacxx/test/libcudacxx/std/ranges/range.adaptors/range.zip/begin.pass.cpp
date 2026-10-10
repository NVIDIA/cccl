//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr auto begin() requires (!(simple-view<Views> && ...));
// constexpr auto begin() const requires (range<const Views> && ...);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

template <class T>
_CCCL_CONCEPT HasConstBegin = _CCCL_REQUIRES_EXPR((T), const T& ct)(void(ct.begin()));

template <class T>
_CCCL_CONCEPT HasBegin = _CCCL_REQUIRES_EXPR((T), T& t)(void(t.begin()));

template <class T>
_CCCL_CONCEPT HasConstAndNonConstBegin = _CCCL_REQUIRES_EXPR((T), T& t, const T& ct)(
  requires(HasConstBegin<T>), requires(!cuda::std::same_as<decltype(t.begin()), decltype(ct.begin())>));

template <class T>
_CCCL_CONCEPT HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
_CCCL_CONCEPT HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

struct NoConstBeginView : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin()
  {
    return nullptr;
  }
  TEST_FUNC int* end()
  {
    return nullptr;
  }
};

TEST_FUNC constexpr bool test()
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
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
