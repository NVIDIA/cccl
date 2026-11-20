//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct NonDefaultConstructible
{
  NonDefaultConstructible() = delete;
};

struct MemberSize
{
  MemberSize() = delete;

  TEST_FUNC constexpr unsigned short size()
  {
    return 42;
  }
};

struct ADLSize
{
  TEST_FUNC friend constexpr int size(ADLSize&)
  {
    return 7;
  }
};

struct RuntimeSize
{
  TEST_FUNC cuda::std::size_t size() const
  {
    return 3;
  }
};

struct ValueDependentSize
{
  cuda::std::size_t count = 0;

  TEST_FUNC constexpr cuda::std::size_t size() const
  {
    return count;
  }
};

struct SentinelSize
{
  int values[3];

  TEST_FUNC constexpr int* begin()
  {
    return values;
  }

  TEST_FUNC constexpr int* end()
  {
    return values + 3;
  }
};

struct DisabledSize : MemberSize
{};

template <>
inline constexpr bool cuda::std::ranges::disable_sized_range<DisabledSize> = true;

struct Identity
{
  TEST_FUNC constexpr int& operator()(int& value) const
  {
    return value;
  }
};

struct IsEven
{
  TEST_FUNC constexpr bool operator()(int value) const
  {
    return value % 2 == 0;
  }
};

struct NonCommonStaticView : cuda::std::ranges::view_base
{
  int* data = nullptr;

  TEST_FUNC constexpr int* begin() const
  {
    return data;
  }

  TEST_FUNC constexpr sentinel_wrapper<int*> end() const
  {
    return sentinel_wrapper<int*>{data + 5};
  }

  TEST_FUNC constexpr cuda::std::size_t size() const
  {
    return 5;
  }
};

// Constness of the referenced range is part of the view's type and can affect its size.
struct CvSizedView : NonCommonStaticView
{
  TEST_FUNC constexpr sentinel_wrapper<int*> end()
  {
    return sentinel_wrapper<int*>{data + 5};
  }

  TEST_FUNC constexpr sentinel_wrapper<int*> end() const
  {
    return sentinel_wrapper<int*>{data + 3};
  }

  TEST_FUNC constexpr cuda::std::size_t size()
  {
    return 5;
  }

  TEST_FUNC constexpr cuda::std::size_t size() const
  {
    return 3;
  }
};

struct CvCommonSizedView : CvSizedView
{
  TEST_FUNC constexpr int* end()
  {
    return data + 5;
  }

  TEST_FUNC constexpr int* end() const
  {
    return data + 3;
  }
};

struct NonCommonDynamicView : NonCommonStaticView
{
  cuda::std::size_t count = 0;

  TEST_FUNC constexpr cuda::std::size_t size() const
  {
    return count;
  }
};

struct UnsizedView : NonCommonStaticView
{};

template <>
inline constexpr bool cuda::std::ranges::disable_sized_range<UnsizedView> = true;

// ranges::size uses an lvalue even when its argument is an rvalue.
struct RefQualifiedSize
{
  TEST_FUNC constexpr int size() &
  {
    return 42;
  }

  TEST_FUNC constexpr int size() &&
  {
    return 7;
  }
};

struct RvalueOnlySize
{
  TEST_FUNC constexpr int size() &&
  {
    return 7;
  }
};

template <class Range, cuda::std::size_t Expected>
TEST_FUNC constexpr void test_references()
{
  static_assert(cuda::std::ranges::__static_size_v<Range> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<Range&> == Expected);
  static_assert(cuda::std::ranges::__static_size_v<Range&&> == Expected);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::__static_size_v<Range>), const cuda::std::size_t>);
}

// References preserve the result; each cv-qualification can select a different size query.
template <class Range,
          cuda::std::size_t Expected,
          cuda::std::size_t ConstExpected         = Expected,
          cuda::std::size_t VolatileExpected      = cuda::std::dynamic_extent,
          cuda::std::size_t ConstVolatileExpected = VolatileExpected>
TEST_FUNC constexpr void test_cvref()
{
  test_references<Range, Expected>();
  test_references<const Range, ConstExpected>();
  test_references<volatile Range, VolatileExpected>();
  test_references<const volatile Range, ConstVolatileExpected>();
}

TEST_FUNC constexpr void test_size_queries()
{
  [[maybe_unused]] constexpr auto dynamic = cuda::std::dynamic_extent;

  // Array extents do not depend on element cv-qualifiers or default construction.
  test_cvref<int[3], 3, 3, 3>();
  test_cvref<int[2][3], 2, 2, 2>();
  test_cvref<NonDefaultConstructible[4], 4, 4, 4>();
  test_cvref<cuda::std::array<int, 0>, 0>();
  test_cvref<cuda::std::array<int, 5>, 5>();
  test_cvref<cuda::std::array<NonDefaultConstructible, 5>, 5>();
  test_cvref<cuda::std::span<int, 0>, 0>();
  test_cvref<cuda::std::span<int, 6>, 6>();
  test_cvref<cuda::std::span<NonDefaultConstructible, 6>, 6>();

  // Member, ADL, and iterator/sentinel queries may only be available for mutable lvalues.
  test_cvref<MemberSize, 42, dynamic>();
  test_cvref<ADLSize, 7, dynamic>();
  test_cvref<SentinelSize, 3, dynamic>();
  test_cvref<RefQualifiedSize, 42, dynamic>();
  test_cvref<RvalueOnlySize, dynamic>();
  test_cvref<CvSizedView, 5, 3>();

  test_cvref<cuda::std::span<int>, dynamic>();
  test_cvref<cuda::std::initializer_list<int>, dynamic>();
  test_cvref<RuntimeSize, dynamic>();
  test_cvref<ValueDependentSize, dynamic>();
  test_cvref<DisabledSize, dynamic>();
  test_cvref<int[], dynamic>();
  test_cvref<int*, dynamic>();
  test_cvref<int, dynamic>();
  static_assert(cuda::std::ranges::__static_size_v<void> == dynamic);
}

TEST_FUNC constexpr void test_wrappers()
{
  namespace ranges                        = cuda::std::ranges;
  [[maybe_unused]] constexpr auto dynamic = cuda::std::dynamic_extent;
  using Array                             = cuda::std::array<int, 5>;
  using Static                            = ranges::ref_view<Array>;
  using Dynamic                           = cuda::std::span<int>;

  // views::all wraps lvalues by reference and owns rvalues without changing their sizes.
  test_cvref<ranges::views::all_t<int (&)[3]>, 3>();
  test_cvref<ranges::views::all_t<Array&>, 5>();
  test_cvref<ranges::views::all_t<const Array&>, 5>();
  test_cvref<ranges::views::all_t<Array>, 5>();
  test_cvref<ranges::views::all_t<Dynamic>, dynamic>();
  test_cvref<ranges::ref_view<Dynamic>, dynamic>();
  test_cvref<ranges::owning_view<Dynamic>, dynamic>();
  test_cvref<ranges::ref_view<ranges::owning_view<Array>>, 5>();
  test_cvref<ranges::owning_view<Static>, 5>();
  test_cvref<ranges::ref_view<ranges::ref_view<int[3]>>, 3>();
  test_cvref<ranges::ref_view<int[2][3]>, 2>();
  test_cvref<ranges::ref_view<cuda::std::array<NonDefaultConstructible, 5>>, 5>();
  test_cvref<ranges::owning_view<cuda::std::array<NonDefaultConstructible, 5>>, 5>();

  // Const ref_view keeps the referenced range's cv-qualifiers, whereas owning_view applies its own.
  test_cvref<ranges::ref_view<CvSizedView>, 5>();
  test_cvref<ranges::ref_view<const CvSizedView>, 3>();
  test_cvref<ranges::owning_view<CvSizedView>, 5, 3>();
  test_cvref<ranges::ref_view<const ranges::ref_view<CvSizedView>>, 5>();
  test_cvref<ranges::ref_view<const ranges::ref_view<const CvSizedView>>, 3>();
  test_cvref<ranges::ref_view<const ranges::owning_view<CvSizedView>>, 3>();
  test_cvref<ranges::ref_view<UnsizedView>, dynamic>();
  test_cvref<ranges::owning_view<UnsizedView>, dynamic>();
}

TEST_FUNC constexpr void test_adaptors()
{
  namespace ranges                        = cuda::std::ranges;
  [[maybe_unused]] constexpr auto dynamic = cuda::std::dynamic_extent;
  using Static                            = ranges::ref_view<cuda::std::array<int, 5>>;
  using Dynamic                           = cuda::std::span<int>;

  test_cvref<ranges::empty_view<int>, 0, 0, 0>();
  test_cvref<ranges::single_view<int>, 1, 1, 1>();

  // Size-preserving adaptors propagate metadata through compositions.
  test_cvref<ranges::transform_view<Static, Identity>, 5>();
  test_cvref<ranges::transform_view<Dynamic, Identity>, dynamic>();
  test_cvref<ranges::reverse_view<Static>, 5>();
  test_cvref<ranges::reverse_view<Dynamic>, dynamic>();
  test_cvref<ranges::reverse_view<ranges::transform_view<Static, Identity>>, 5>();
  test_cvref<ranges::transform_view<ranges::reverse_view<Static>, Identity>, 5>();
  test_cvref<ranges::common_view<NonCommonStaticView>, 5>();
  test_cvref<ranges::common_view<NonCommonDynamicView>, dynamic>();
  test_cvref<ranges::ref_view<ranges::common_view<NonCommonStaticView>>, 5>();

  test_cvref<ranges::transform_view<CvSizedView, Identity>, 5, 3>();
  test_cvref<ranges::reverse_view<CvCommonSizedView>, 5, 3>();
  test_cvref<ranges::common_view<CvSizedView>, 5, 3>();
  test_cvref<ranges::ref_view<const ranges::transform_view<CvSizedView, Identity>>, 3>();
  test_cvref<ranges::ref_view<const ranges::reverse_view<CvCommonSizedView>>, 3>();
  test_cvref<ranges::ref_view<const ranges::common_view<CvSizedView>>, 3>();

  // Unsized bases and runtime counts or predicates must not expose a nonzero static size.
  test_cvref<ranges::transform_view<UnsizedView, Identity>, dynamic>();
  test_cvref<ranges::reverse_view<UnsizedView>, dynamic>();
  test_cvref<ranges::common_view<UnsizedView>, dynamic>();
  test_cvref<ranges::take_view<Static>, dynamic>();
  test_cvref<ranges::drop_view<Static>, dynamic>();
  test_cvref<ranges::take_view<Dynamic>, dynamic>();
  test_cvref<ranges::drop_view<Dynamic>, dynamic>();
  test_cvref<ranges::filter_view<Static, IsEven>, dynamic>();
  test_cvref<ranges::take_while_view<Static, IsEven>, dynamic>();
  test_cvref<ranges::drop_while_view<Static, IsEven>, dynamic>();
  test_cvref<ranges::ref_view<ranges::take_view<Static>>, dynamic>();
  test_cvref<ranges::owning_view<ranges::drop_view<Static>>, dynamic>();
}

TEST_FUNC constexpr void test_zip()
{
  namespace ranges                        = cuda::std::ranges;
  [[maybe_unused]] constexpr auto dynamic = cuda::std::dynamic_extent;
  using Static                            = ranges::ref_view<cuda::std::array<int, 5>>;
  using Dynamic                           = cuda::std::span<int>;

  test_cvref<decltype(ranges::views::zip()), 0, 0, 0>();
  test_cvref<ranges::zip_view<Static>, 5>();
  test_cvref<ranges::zip_view<Static, cuda::std::span<int, 3>>, 3>();
  test_cvref<ranges::zip_view<cuda::std::span<int, 3>, Static>, 3>();
  test_cvref<ranges::zip_view<Static, Static, cuda::std::span<int, 7>>, 5>();
  test_cvref<ranges::zip_view<Static, Dynamic>, dynamic>();
  test_cvref<ranges::zip_view<Dynamic, Static>, dynamic>();
  test_cvref<ranges::zip_view<Static, UnsizedView>, dynamic>();
  test_cvref<ranges::reverse_view<ranges::zip_view<Static, cuda::std::span<int, 3>>>, 3>();
  test_cvref<ranges::zip_view<CvSizedView, Static>, 5, 3>();
  test_cvref<ranges::ref_view<const ranges::zip_view<CvSizedView, Static>>, 3>();
}

TEST_FUNC constexpr void test_empty_views()
{
  namespace ranges                        = cuda::std::ranges;
  [[maybe_unused]] constexpr auto dynamic = cuda::std::dynamic_extent;
  using Empty                             = ranges::ref_view<cuda::std::array<int, 0>>;
  using Dynamic                           = cuda::std::span<int>;
  using Filter                            = ranges::filter_view<Empty, IsEven>;
  using TakeWhile                         = ranges::take_while_view<Empty, IsEven>;
  using DropWhile                         = ranges::drop_while_view<Empty, IsEven>;
  using Infinite                          = ranges::iota_view<int>;

  // Cardinality can be known even when ranges::size is unavailable.
  static_assert(!ranges::sized_range<Filter>);
  static_assert(!ranges::sized_range<TakeWhile>);
  static_assert(!ranges::sized_range<ranges::drop_while_view<TakeWhile, IsEven>>);
  test_cvref<Filter, 0, dynamic>();
  test_cvref<TakeWhile, 0>();
  test_cvref<DropWhile, 0, dynamic>();
  test_cvref<ranges::ref_view<const TakeWhile>, 0>();

  test_cvref<ranges::zip_view<Empty, Dynamic>, 0>();
  test_cvref<ranges::zip_view<Dynamic, Empty>, 0>();
  test_cvref<ranges::zip_view<Empty, UnsizedView>, 0>();
  test_cvref<ranges::zip_view<UnsizedView, Empty>, 0>();
  test_cvref<ranges::zip_view<Empty, Infinite>, 0>();
  test_cvref<ranges::zip_view<Infinite, Empty>, 0>();
  test_cvref<ranges::zip_view<Filter, Dynamic>, 0, dynamic>();
  test_cvref<ranges::zip_view<UnsizedView, Infinite>, dynamic>();
  test_cvref<Infinite, dynamic>();

  // Const views without iteration support have no static size.
  static_assert(!ranges::range<const Filter>);
  test_cvref<ranges::ref_view<Filter>, 0>();
  test_cvref<ranges::owning_view<Filter>, 0, dynamic>();
  test_cvref<ranges::transform_view<Filter, Identity>, 0, dynamic>();
  test_cvref<ranges::reverse_view<Filter>, 0, dynamic>();
  test_cvref<ranges::common_view<TakeWhile>, 0>();
  test_cvref<ranges::take_view<Filter>, 0, dynamic>();
  test_cvref<ranges::drop_view<Filter>, 0, dynamic>();
  test_cvref<ranges::filter_view<DropWhile, IsEven>, 0, dynamic>();
  test_cvref<ranges::take_while_view<Filter, IsEven>, 0, dynamic>();
  test_cvref<ranges::drop_while_view<TakeWhile, IsEven>, 0, dynamic>();
  test_cvref<ranges::reverse_view<ranges::zip_view<Empty, UnsizedView>>, 0, dynamic>();
  test_cvref<ranges::zip_view<Filter, Empty>, 0, dynamic>();

  test_cvref<ranges::take_view<Empty>, 0>();
  test_cvref<ranges::drop_view<Empty>, 0>();
  test_cvref<ranges::ref_view<const ranges::take_view<Empty>>, 0>();
  test_cvref<ranges::ref_view<const ranges::drop_view<Empty>>, 0>();
  test_cvref<ranges::reverse_view<ranges::take_view<Empty>>, 0>();
}

TEST_FUNC constexpr bool test()
{
  test_size_queries();
  test_wrappers();
  test_adaptors();
  test_zip();
  test_empty_views();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
