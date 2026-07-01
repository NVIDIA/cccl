//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// libstdc++ < 10 uses some of the names we treat as nasty macros. Let's omit the "nasty_macros.h" inclusion for this
// test.
// ADDITIONAL_COMPILE_DEFINITIONS: SUPPORT_NASTY_MACROS_H

// <cuda/std/format>

// template<ranges::input_range R>
//     requires same_as<R, remove_cvref_t<R>>
//  constexpr range_format format_kind<R> = see below;

#include "test_macros.h"

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#  include <deque>
#  if __has_include(<filesystem>)
#    include <filesystem>
#  endif // __has_include(<filesystem>)
#  include <forward_list>
#  include <list>
#  include <map>
#  include <set>
#  include <unordered_map>
#  include <unordered_set>
#  include <valarray>
#  include <vector>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include <cuda/std/__format_>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/span>

// [format.range.fmtkind]
// If same_as<remove_cvref_t<ranges::range_reference_t<R>>, R> is true,
// format_kind<R> is range_format::disabled.
// [Note 1: This prevents constraint recursion for ranges whose reference type
// is the same range type. For example, cuda::std::filesystem::path is a range of
// cuda::std::filesystem::path. - end note]
struct recursive_range
{
  struct iterator
  {
    using iterator_concept = cuda::std::input_iterator_tag;
    using value_type       = recursive_range;
    using difference_type  = cuda::std::ptrdiff_t;
    using reference        = recursive_range;

    TEST_FUNC reference operator*() const;

    TEST_FUNC iterator& operator++();
    TEST_FUNC iterator operator++(int);

    TEST_FUNC friend bool operator==(const iterator&, const iterator&);
    TEST_FUNC friend bool operator!=(const iterator&, const iterator&);
  };

  TEST_FUNC iterator begin();
  TEST_FUNC iterator end();
};

static_assert(cuda::std::ranges::input_range<recursive_range>, "format_kind requires an input range");
static_assert(cuda::std::format_kind<recursive_range> == cuda::std::range_format::disabled);

static_assert(cuda::std::format_kind<cuda::std::array<int, 1>> == cuda::std::range_format::sequence);
static_assert(cuda::std::format_kind<cuda::std::span<int>> == cuda::std::range_format::sequence);
static_assert(cuda::std::format_kind<cuda::std::inplace_vector<int, 1>> == cuda::std::range_format::sequence);

#if _CCCL_HAS_HOST_STD_LIB()
#  if __has_include(<filesystem>)
static_assert(cuda::std::format_kind<std::filesystem::path> == cuda::std::range_format::disabled);
#  endif // __has_include(<filesystem>)

static_assert(cuda::std::format_kind<std::map<int, int>> == cuda::std::range_format::map);
static_assert(cuda::std::format_kind<std::multimap<int, int>> == cuda::std::range_format::map);
static_assert(cuda::std::format_kind<std::unordered_map<int, int>> == cuda::std::range_format::map);
static_assert(cuda::std::format_kind<std::unordered_multimap<int, int>> == cuda::std::range_format::map);

static_assert(cuda::std::format_kind<std::set<int>> == cuda::std::range_format::set);
static_assert(cuda::std::format_kind<std::multiset<int>> == cuda::std::range_format::set);
static_assert(cuda::std::format_kind<std::unordered_set<int>> == cuda::std::range_format::set);
static_assert(cuda::std::format_kind<std::unordered_multiset<int>> == cuda::std::range_format::set);

static_assert(cuda::std::format_kind<std::array<int, 1>> == cuda::std::range_format::sequence);
static_assert(cuda::std::format_kind<std::vector<int>> == cuda::std::range_format::sequence);
static_assert(cuda::std::format_kind<std::deque<int>> == cuda::std::range_format::sequence);
static_assert(cuda::std::format_kind<std::forward_list<int>> == cuda::std::range_format::sequence);
static_assert(cuda::std::format_kind<std::list<int>> == cuda::std::range_format::sequence);

static_assert(cuda::std::format_kind<std::valarray<int>> == cuda::std::range_format::sequence);
#endif // _CCCL_HAS_HOST_STD_LIB()

// [format.range.fmtkind]/3
//   Remarks: Pursuant to [namespace.std], users may specialize format_kind for
//   cv-unqualified program-defined types that model ranges::input_range. Such
//   specializations shall be usable in constant expressions ([expr.const]) and
//   have type const range_format.
// Note only test the specializing, not all constraints.
struct no_specialization : cuda::std::ranges::view_base
{
  using key_type = void;
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};
static_assert(cuda::std::format_kind<no_specialization> == cuda::std::range_format::set);

// The struct's "contents" are the same as no_specialization.
struct specialized : cuda::std::ranges::view_base
{
  using key_type = void;
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

template <>
constexpr cuda::std::range_format cuda::std::format_kind<specialized> = cuda::std::range_format::sequence;
static_assert(cuda::std::format_kind<specialized> == cuda::std::range_format::sequence);

int main(int, char**)
{
  return 0;
}
