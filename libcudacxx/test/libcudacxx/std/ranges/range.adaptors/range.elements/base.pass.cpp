//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr V base() const & requires copy_constructible<V> { return base_; }
// constexpr V base() && { return cuda::std::move(base_); }

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct View : cuda::std::ranges::view_base
{
  int i;
  __host__ __device__ cuda::std::tuple<int>* begin() const;
  __host__ __device__ cuda::std::tuple<int>* end() const;
};

struct MoveOnlyView : View
{
  MoveOnly mo;
};

template <class T>
_CCCL_CONCEPT HasBase = _CCCL_REQUIRES_EXPR((T), T&& t)((cuda::std::forward<T>(t).base()));

static_assert(HasBase<cuda::std::ranges::elements_view<View, 0> const&>);
static_assert(HasBase<cuda::std::ranges::elements_view<View, 0>&&>);

static_assert(!HasBase<cuda::std::ranges::elements_view<MoveOnlyView, 0> const&>);
static_assert(HasBase<cuda::std::ranges::elements_view<MoveOnlyView, 0>&&>);

__host__ __device__ constexpr bool test()
{
  // const &
  {
    const cuda::std::ranges::elements_view<View, 0> ev{View{{}, 5}};
    decltype(auto) v = ev.base();
    static_assert(cuda::std::same_as<decltype(v), View>);
    assert(v.i == 5);
  }

  // &
  {
    cuda::std::ranges::elements_view<View, 0> ev{View{{}, 5}};
    decltype(auto) v = ev.base();
    static_assert(cuda::std::same_as<decltype(v), View>);
    assert(v.i == 5);
  }

  // &&
  {
    cuda::std::ranges::elements_view<View, 0> ev{View{{}, 5}};
    decltype(auto) v = cuda::std::move(ev).base();
    static_assert(cuda::std::same_as<decltype(v), View>);
    assert(v.i == 5);
  }

  // const &&
  {
    const cuda::std::ranges::elements_view<View, 0> ev{View{{}, 5}};
    decltype(auto) v = cuda::std::move(ev).base();
    static_assert(cuda::std::same_as<decltype(v), View>);
    assert(v.i == 5);
  }

  // move only
  {
    cuda::std::ranges::elements_view<MoveOnlyView, 0> ev{MoveOnlyView{{}, 5}};
    decltype(auto) v = cuda::std::move(ev).base();
    static_assert(cuda::std::same_as<decltype(v), MoveOnlyView>);
    assert(v.mo.get() == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
