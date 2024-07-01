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

// template<range R>
//  requires is_object_v<R>
// class ref_view;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

__device__ int globalBuff[8];

#if TEST_STD_VER >= 2020
template <class T>
concept ValidRefView = requires { typename cuda::std::ranges::ref_view<T>; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
constexpr bool ValidRefView = false;
template <class T>
constexpr bool ValidRefView<T, cuda::std::void_t<cuda::std::ranges::ref_view<T>>> = true;
#endif // TEST_STD_VER <= 2017

struct Range
{
  int start = 0;
  __host__ __device__ friend constexpr int* begin(Range const& range)
  {
    return globalBuff + range.start;
  }
  __host__ __device__ friend constexpr int* end(Range const&)
  {
    return globalBuff + 8;
  }
  __host__ __device__ friend constexpr int* begin(Range& range)
  {
    return globalBuff + range.start;
  }
  __host__ __device__ friend constexpr int* end(Range&)
  {
    return globalBuff + 8;
  }
};

struct BeginOnly
{
  __host__ __device__ friend int* begin(BeginOnly const&);
  __host__ __device__ friend int* begin(BeginOnly&);
};

static_assert(ValidRefView<Range>);
static_assert(!ValidRefView<BeginOnly>);
static_assert(!ValidRefView<int (&)[4]>);
static_assert(ValidRefView<int[4]>);

static_assert(cuda::std::derived_from<cuda::std::ranges::ref_view<Range>,
                                      cuda::std::ranges::view_interface<cuda::std::ranges::ref_view<Range>>>);

struct RangeConvertible
{
  __host__ __device__ operator Range&();
};

struct RValueRangeConvertible
{
  __host__ __device__ operator Range&&();
};

static_assert(cuda::std::is_constructible_v<cuda::std::ranges::ref_view<Range>, Range&>);
static_assert(cuda::std::is_constructible_v<cuda::std::ranges::ref_view<Range>, RangeConvertible>);
static_assert(!cuda::std::is_constructible_v<cuda::std::ranges::ref_view<Range>, RValueRangeConvertible>);

struct ConstConvertibleToLValueAndRValue
{
  __host__ __device__ operator Range&() const;
  __host__ __device__ operator Range&&() const;
};
static_assert(cuda::std::is_convertible_v<RangeConvertible, cuda::std::ranges::ref_view<Range>>);
static_assert(!cuda::std::is_convertible_v<RValueRangeConvertible, cuda::std::ranges::ref_view<Range>>);
static_assert(!cuda::std::is_convertible_v<ConstConvertibleToLValueAndRValue, cuda::std::ranges::ref_view<Range>>);

struct ForwardRange
{
  __host__ __device__ constexpr forward_iterator<int*> begin() const
  {
    return forward_iterator<int*>(globalBuff);
  }
  __host__ __device__ constexpr forward_iterator<int*> end() const
  {
    return forward_iterator<int*>(globalBuff + 8);
  }
};

struct Cpp17InputRange
{
  struct sentinel
  {
    __host__ __device__ friend constexpr bool operator==(sentinel, cpp17_input_iterator<int*> iter)
    {
      return base(iter) == globalBuff + 8;
    }
#if TEST_STD_VER <= 2017
    __host__ __device__ friend constexpr bool operator==(cpp17_input_iterator<int*> iter, sentinel)
    {
      return base(iter) == globalBuff + 8;
    }
    __host__ __device__ friend constexpr bool operator!=(sentinel, cpp17_input_iterator<int*> iter)
    {
      return base(iter) != globalBuff + 8;
    }
    __host__ __device__ friend constexpr bool operator!=(cpp17_input_iterator<int*> iter, sentinel)
    {
      return base(iter) != globalBuff + 8;
    }
#endif // TEST_STD_VER <= 2017
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(sentinel, cpp17_input_iterator<int*>)
    {
      return -8;
    }
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(cpp17_input_iterator<int*>, sentinel)
    {
      return 8;
    }
  };

  __host__ __device__ constexpr cpp17_input_iterator<int*> begin() const
  {
    return cpp17_input_iterator<int*>(globalBuff);
  }
  __host__ __device__ constexpr sentinel end() const
  {
    return {};
  }
};

struct Cpp20InputRange
{
  struct sentinel
  {
    __host__ __device__ friend constexpr bool operator==(sentinel, const cpp20_input_iterator<int*>& iter)
    {
      return base(iter) == globalBuff + 8;
    }
#if TEST_STD_VER <= 2017
    __host__ __device__ friend constexpr bool operator==(const cpp20_input_iterator<int*>& iter, sentinel)
    {
      return base(iter) == globalBuff + 8;
    }
    __host__ __device__ friend constexpr bool operator!=(sentinel, const cpp20_input_iterator<int*>& iter)
    {
      return base(iter) != globalBuff + 8;
    }
    __host__ __device__ friend constexpr bool operator!=(const cpp20_input_iterator<int*>& iter, sentinel)
    {
      return base(iter) != globalBuff + 8;
    }
#endif // TEST_STD_VER <= 2017
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(sentinel, const cpp20_input_iterator<int*>&)
    {
      return -8;
    }
  };

  __host__ __device__ constexpr cpp20_input_iterator<int*> begin() const
  {
    return cpp20_input_iterator<int*>(globalBuff);
  }
  __host__ __device__ constexpr sentinel end() const
  {
    return {};
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<Cpp20InputRange> = true;

#if TEST_STD_VER >= 2020
template <class R>
concept EmptyIsInvocable = requires(cuda::std::ranges::ref_view<R> view) { view.empty(); };

template <class R>
concept SizeIsInvocable = requires(cuda::std::ranges::ref_view<R> view) { view.size(); };

template <class R>
concept DataIsInvocable = requires(cuda::std::ranges::ref_view<R> view) { view.data(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class R>
_LIBCUDACXX_CONCEPT_FRAGMENT(EmptyIsInvocable_, requires(cuda::std::ranges::ref_view<R> view)((view.empty())));
template <class R>
_LIBCUDACXX_CONCEPT EmptyIsInvocable = _LIBCUDACXX_FRAGMENT(EmptyIsInvocable_, R);

template <class R>
_LIBCUDACXX_CONCEPT_FRAGMENT(SizeIsInvocable_, requires(cuda::std::ranges::ref_view<R> view)((view.size())));
template <class R>
_LIBCUDACXX_CONCEPT SizeIsInvocable = _LIBCUDACXX_FRAGMENT(SizeIsInvocable_, R);

template <class R>
_LIBCUDACXX_CONCEPT_FRAGMENT(DataIsInvocable_, requires(cuda::std::ranges::ref_view<R> view)((view.data())));
template <class R>
_LIBCUDACXX_CONCEPT DataIsInvocable = _LIBCUDACXX_FRAGMENT(DataIsInvocable_, R);
#endif // TEST_STD_VER <= 2017

// Testing ctad.
static_assert(cuda::std::same_as<decltype(cuda::std::ranges::ref_view(cuda::std::declval<Range&>())),
                                 cuda::std::ranges::ref_view<Range>>);

__host__ __device__ constexpr bool test()
{
  {
    // ref_view::base
    Range range{};
    cuda::std::ranges::ref_view view{range};
    assert(view.begin() == globalBuff);
    view.base() = Range{2};
    assert(view.begin() == globalBuff + 2);
  }

  {
    // ref_view::begin
    Range range1{};
    cuda::std::ranges::ref_view view1 = range1;
    assert(view1.begin() == globalBuff);

    ForwardRange range2{};
    cuda::std::ranges::ref_view view2 = range2;
    assert(base(view2.begin()) == globalBuff);

    Cpp17InputRange range3{};
    cuda::std::ranges::ref_view view3 = range3;
    assert(base(view3.begin()) == globalBuff);

    Cpp20InputRange range4{};
    cuda::std::ranges::ref_view view4 = range4;
    assert(base(view4.begin()) == globalBuff);
  }

  {
    // ref_view::end
    Range range1{};
    cuda::std::ranges::ref_view view1 = range1;
    assert(view1.end() == globalBuff + 8);

    ForwardRange range2{};
    cuda::std::ranges::ref_view view2 = range2;
    assert(base(view2.end()) == globalBuff + 8);

    Cpp17InputRange range3{};
    cuda::std::ranges::ref_view view3 = range3;
    assert(view3.end() == cpp17_input_iterator<int*>(globalBuff + 8));

    Cpp20InputRange range4{};
    cuda::std::ranges::ref_view view4 = range4;
    assert(view4.end() == cpp20_input_iterator<int*>(globalBuff + 8));
  }

  {
    // ref_view::empty
    Range range{8};
    cuda::std::ranges::ref_view view1 = range;
    assert(view1.empty());

    ForwardRange range2{};
    cuda::std::ranges::ref_view view2 = range2;
    assert(!view2.empty());

    static_assert(!EmptyIsInvocable<Cpp17InputRange>);
    static_assert(!EmptyIsInvocable<Cpp20InputRange>);
  }

  {
    // ref_view::size
    Range range1{8};
    cuda::std::ranges::ref_view view1 = range1;
    assert(view1.size() == 0);

    Range range2{2};
    cuda::std::ranges::ref_view view2 = range2;
    assert(view2.size() == 6);

    static_assert(!SizeIsInvocable<ForwardRange>);
  }

  {
    // ref_view::data
    Range range1{};
    cuda::std::ranges::ref_view view1 = range1;
    assert(view1.data() == globalBuff);

    Range range2{2};
    cuda::std::ranges::ref_view view2 = range2;
    assert(view2.data() == globalBuff + 2);

    static_assert(!DataIsInvocable<ForwardRange>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // _LIBCUDACXX_ADDRESSOF

  return 0;
}
