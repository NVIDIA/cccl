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

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "types.h"

#if TEST_STD_VER >= 2020
template <class T>
concept HasSize = requires(T t) { t.size(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
constexpr bool HasSize = false;

template <class T>
constexpr bool HasSize<T, cuda::std::void_t<decltype(cuda::std::declval<T>().size())>> = true;
#endif // TEST_STD_VER <= 2017
static_assert(HasSize<cuda::std::ranges::elements_view<SimpleCommon, 0>>);
static_assert(HasSize<const cuda::std::ranges::elements_view<SimpleCommon, 0>>);

struct NonSized : cuda::std::ranges::view_base
{
  using iterator = forward_iterator<cuda::std::tuple<int>*>;
  __host__ __device__ iterator begin() const;
  __host__ __device__ iterator end() const;
};
static_assert(!cuda::std::ranges::sized_range<NonSized>);
static_assert(!cuda::std::ranges::sized_range<const NonSized>);

static_assert(!HasSize<cuda::std::ranges::elements_view<NonSized, 0>>);
static_assert(!HasSize<const cuda::std::ranges::elements_view<NonSized, 0>>);

struct SizedNonConst : TupleBufferView
{
  DELEGATE_TUPLEBUFFERVIEW(SizedNonConst)

  using iterator = forward_iterator<cuda::std::tuple<int>*>;
  __host__ __device__ constexpr auto begin() const
  {
    return iterator{buffer_};
  }
  __host__ __device__ constexpr auto end() const
  {
    return iterator{buffer_ + size_};
  }
  __host__ __device__ constexpr cuda::std::size_t size()
  {
    return size_;
  }
};

static_assert(HasSize<cuda::std::ranges::elements_view<SizedNonConst, 0>>);
static_assert(!HasSize<const cuda::std::ranges::elements_view<SizedNonConst, 0>>);

struct OnlyConstSized : TupleBufferView
{
  DELEGATE_TUPLEBUFFERVIEW(OnlyConstSized)

  using iterator = forward_iterator<cuda::std::tuple<int>*>;
  __host__ __device__ constexpr auto begin() const
  {
    return iterator{buffer_};
  }
  __host__ __device__ constexpr auto end() const
  {
    return iterator{buffer_ + size_};
  }
  __host__ __device__ constexpr cuda::std::size_t size() const
  {
    return size_;
  }
  __host__ __device__ constexpr cuda::std::size_t size() = delete;
};

static_assert(HasSize<const OnlyConstSized>);
static_assert(HasSize<cuda::std::ranges::elements_view<OnlyConstSized, 0>>);
static_assert(HasSize<const cuda::std::ranges::elements_view<OnlyConstSized, 0>>);

__host__ __device__ constexpr bool test()
{
  cuda::std::tuple<int> buffer[] = {{1}, {2}, {3}};

  // non-const and const are sized
  {
    auto ev = cuda::std::views::elements<0>(buffer);
    assert(ev.size() == 3);
    assert(cuda::std::as_const(ev).size() == 3);
  }

  {
    // const-view non-sized range
    auto ev = cuda::std::views::elements<0>(SizedNonConst{buffer});
    assert(ev.size() == 3);
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
