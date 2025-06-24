//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

//  constexpr auto begin() requires (!simple-view<V>)
//  constexpr auto begin() const requires range<const V>

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

template <class View>
_CCCL_CONCEPT HasConstBegin = _CCCL_REQUIRES_EXPR((View), const View cv)((cv.begin()));

template <class View>
_CCCL_CONCEPT HasBegin = _CCCL_REQUIRES_EXPR((View), View v)((v.begin()));

// because const begin and non-const begin returns different types (iterator<true>, iterator<false>)
template <class View>
_CCCL_CONCEPT HasConstAndNonConstBegin = _CCCL_REQUIRES_EXPR((View), View v, const View cv)(
  requires(HasConstBegin<View>), requires(!cuda::std::same_as<decltype(v.begin()), decltype(cv.begin())>));

template <class View>
_CCCL_CONCEPT HasOnlyNonConstBegin =
  _CCCL_REQUIRES_EXPR((View))(requires(HasBegin<View>), requires(!HasConstBegin<View>));

template <class View>
_CCCL_CONCEPT HasOnlyConstBegin = _CCCL_REQUIRES_EXPR((View))(requires(!HasBegin<View>), requires(HasConstBegin<View>));

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
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
