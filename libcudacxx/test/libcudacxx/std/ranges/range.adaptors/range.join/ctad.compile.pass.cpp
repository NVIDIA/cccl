//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// template<class R>
//   explicit join_view(R&&) -> join_view<views::all_t<R>>;

#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Child
{
  TEST_FUNC constexpr int* begin() const
  {
    return nullptr;
  }
  TEST_FUNC constexpr int* end() const
  {
    return nullptr;
  }
};

struct View : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr Child* begin() const
  {
    return nullptr;
  }
  TEST_FUNC constexpr Child* end() const
  {
    return nullptr;
  }
};

struct Range
{
  TEST_FUNC constexpr Child* begin() const
  {
    return nullptr;
  }
  TEST_FUNC constexpr Child* end() const
  {
    return nullptr;
  }
};

struct BorrowedRange
{
  TEST_FUNC constexpr Child* begin() const
  {
    return nullptr;
  }
  TEST_FUNC constexpr Child* end() const
  {
    return nullptr;
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowedRange> = true;

struct NestedChildren : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr View* begin() const
  {
    return nullptr;
  }
  TEST_FUNC constexpr View* end() const
  {
    return nullptr;
  }
};

// GCC really does not like local type defs...
using result_join_view                 = cuda::std::ranges::join_view<View>;
using result_join_view_ref             = cuda::std::ranges::join_view<cuda::std::ranges::ref_view<Range>>;
using result_join_view_ref_borrowed    = cuda::std::ranges::join_view<cuda::std::ranges::ref_view<BorrowedRange>>;
using result_join_view_owning          = cuda::std::ranges::join_view<cuda::std::ranges::owning_view<Range>>;
using result_join_view_owning_borrowed = cuda::std::ranges::join_view<cuda::std::ranges::owning_view<BorrowedRange>>;

TEST_FUNC constexpr bool test()
{
  View v{};
  Range r{};
  BorrowedRange br{};

  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::join_view(v)), result_join_view>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::join_view(cuda::std::move(v))), result_join_view>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::join_view(r)), result_join_view_ref>);
  static_assert(
    cuda::std::same_as<decltype(cuda::std::ranges::join_view(cuda::std::move(r))), result_join_view_owning>);
  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::join_view(br)), result_join_view_ref_borrowed>);
  static_assert(
    cuda::std::same_as<decltype(cuda::std::ranges::join_view(cuda::std::move(br))), result_join_view_owning_borrowed>);
  unused(v);
  unused(r);
  unused(br);

  NestedChildren n{};
  cuda::std::ranges::join_view jv(n);

  // CTAD generated from the copy constructor instead of joining the join_view
  auto view = cuda::std::ranges::join_view(jv);
  static_assert(cuda::std::same_as<decltype(view), decltype(jv)>);
  unused(view);

  // CTAD generated from the move constructor instead of joining the join_view
  auto view2 = cuda::std::ranges::join_view(cuda::std::move(jv));
  static_assert(cuda::std::same_as<decltype(view2), decltype(jv)>);
  unused(view2);
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
