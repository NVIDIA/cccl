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

// constexpr const iterator_t<Base>& base() const & noexcept;
// constexpr iterator_t<Base> base() &&;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../types.h"
#include "MoveOnly.h"
#include "test_macros.h"

// Test Noexcept
#if TEST_STD_VER >= 2020
template <class T>
concept IsBaseNoexcept = requires {
  { cuda::std::declval<T>().base() } noexcept;
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool IsBaseNoexcept = false;

template <class T>
inline constexpr bool IsBaseNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T>().base())>> =
  noexcept(cuda::std::declval<T>().base());
#endif // TEST_STD_VER <= 2017
using BaseView     = cuda::std::ranges::subrange<cuda::std::tuple<int>*>;
using ElementsIter = cuda::std::ranges::iterator_t<cuda::std::ranges::elements_view<BaseView, 0>>;

static_assert(IsBaseNoexcept<const ElementsIter&>);
static_assert(IsBaseNoexcept<ElementsIter&>);
static_assert(IsBaseNoexcept<const ElementsIter&&>);
#if !defined(TEST_COMPILER_ICC) // broken noexcept
static_assert(!IsBaseNoexcept<ElementsIter&&>);
#endif // TEST_COMPILER_ICC

struct MoveOnlyIter : IterBase<MoveOnlyIter>
{
  MoveOnly mo;
};
struct Sent
{
  __host__ __device__ friend constexpr bool operator==(const Sent&, const MoveOnlyIter&)
  {
    return true;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ friend constexpr bool operator==(const MoveOnlyIter&, const Sent&)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(const Sent&, const MoveOnlyIter&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator!=(const MoveOnlyIter&, const Sent&)
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

__host__ __device__ constexpr bool test()
{
  cuda::std::tuple<int> t{5};

  // const &
  {
    const ElementsIter it{&t};
    decltype(auto) base = it.base();
    static_assert(cuda::std::is_same_v<decltype(base), cuda::std::tuple<int>* const&>);
    assert(base == &t);
  }

  // &
  {
    ElementsIter it{&t};
    decltype(auto) base = it.base();
    static_assert(cuda::std::is_same_v<decltype(base), cuda::std::tuple<int>* const&>);
    assert(base == &t);
  }

  // &&
  {
    ElementsIter it{&t};
    decltype(auto) base = cuda::std::move(it).base();
    static_assert(cuda::std::is_same_v<decltype(base), cuda::std::tuple<int>*>);
    assert(base == &t);
  }

  // const &&
  {
    const ElementsIter it{&t};
    decltype(auto) base = cuda::std::move(it).base();
    static_assert(cuda::std::is_same_v<decltype(base), cuda::std::tuple<int>* const&>);
    assert(base == &t);
  }

  // move only
  {
    using MoveOnlyElemIter =
      cuda::std::ranges::iterator_t<cuda::std::ranges::elements_view<cuda::std::ranges::subrange<MoveOnlyIter, Sent>, 0>>;

    MoveOnlyElemIter it{MoveOnlyIter{{}, MoveOnly{5}}};
    decltype(auto) base = cuda::std::move(it).base();
    static_assert(cuda::std::is_same_v<decltype(base), MoveOnlyIter>);
    assert(base.mo.get() == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
