//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TRANSFORM_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TRANSFORM_TYPES_H

#include <cuda/std/functional>
#include <cuda/std/ranges>

#include "../range_adaptor_types.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

struct IntView : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

struct MakeTuple
{
  template <class... T>
  TEST_FUNC constexpr auto operator()(T&&... args) const
  {
    return cuda::std::tuple(cuda::std::forward<decltype(args)>(args)...);
  }
};

struct Tie
{
  template <class... T>
  TEST_FUNC constexpr auto operator()(T&&... args) const
  {
    return cuda::std::tie(cuda::std::forward<decltype(args)>(args)...);
  }
};

struct GetFirst
{
  template <class U, class... T>
  TEST_FUNC constexpr decltype(auto) operator()(U&& first, T&&...) const
  {
    return cuda::std::forward<decltype(first)>(first);
  }
};

struct NoConstBeginView : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin();
  TEST_FUNC int* end();
};

struct ConstNonConstDifferentView : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin();
  TEST_FUNC const int* begin() const;
  TEST_FUNC int* end();
  TEST_FUNC const int* end() const;
};

struct NonConstOnlyFn
{
  TEST_FUNC int operator()(int&) const;
  TEST_FUNC int operator()(const int&) const = delete;
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ZIP_TRANSFORM_TYPES_H
