//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/functional>

#include <cuda/experimental/__utility/unstable_unique.cuh>

#include <algorithm>
#include <iterator>
#include <list>
#include <vector>

#include <c2h/catch2_test_helper.h>

namespace cudax = cuda::experimental;

TEST_CASE("unstable_unique empty range", "[utility]")
{
  std::vector<int> v;
  auto new_end = cudax::unstable_unique(v.begin(), v.end());
  REQUIRE(v.end() == new_end);
}

TEST_CASE("unstable_unique no duplicates", "[utility]")
{
  std::vector<int> v = {1, 2, 3, 4, 5};
  auto new_end       = cudax::unstable_unique(v.begin(), v.end());
  REQUIRE(v.end() == new_end);
  REQUIRE(std::vector<int>({1, 2, 3, 4, 5}) == v);
}

TEST_CASE("unstable_unique leading duplicates", "[utility]")
{
  std::vector<int> v = {1, 1, 2, 3, 4, 5};
  auto new_end       = cudax::unstable_unique(v.begin(), v.end());
  REQUIRE(v.begin() + 5 == new_end);
  REQUIRE(std::vector<int>({1, 5, 2, 3, 4, 5}) == v);
}

TEST_CASE("unstable_unique interleaved duplicates", "[utility]")
{
  std::vector<int> v = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  auto new_end       = cudax::unstable_unique(v.begin(), v.end());
  REQUIRE(v.begin() + 5 == new_end);
  REQUIRE(std::vector<int>({1, 5, 2, 4, 3, 3, 4, 4, 5, 5}) == v);
}

TEST_CASE("unstable_unique all same", "[utility]")
{
  std::vector<int> v = {1, 1, 1, 1, 1};
  auto new_end       = cudax::unstable_unique(v.begin(), v.end());
  REQUIRE(1 + v.begin() == new_end);
  REQUIRE(std::vector<int>({1, 1, 1, 1, 1}) == v);
}

TEST_CASE("unstable_unique trailing unique", "[utility]")
{
  std::vector<int> v = {1, 1, 1, 1, 1, 2};
  auto new_end       = cudax::unstable_unique(v.begin(), v.end());
  REQUIRE(v.begin() + 2 == new_end);
  REQUIRE(std::vector<int>({1, 2, 1, 1, 1, 2}) == v);
}

TEST_CASE("unstable_unique with custom predicate", "[utility]")
{
  std::vector<int> v = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5};
  auto new_end       = cudax::unstable_unique(v.begin(), v.end(), cuda::std::equal_to<int>{});
  REQUIRE(v.begin() + 5 == new_end);
  REQUIRE(std::vector<int>{1, 5, 4, 3, 2, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5} == v);
}

TEST_CASE("unstable_unique on bidirectional iterators (std::list)", "[utility]")
{
  // std::list has bidirectional (not random-access) iterators -- exercises the
  // !=-based loop termination path.
  std::list<int> l = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  auto new_end     = cudax::unstable_unique(l.begin(), l.end());
  REQUIRE(std::distance(l.begin(), new_end) == 5);

  std::vector<int> deduped(l.begin(), new_end);
  std::sort(deduped.begin(), deduped.end());
  REQUIRE(std::vector<int>({1, 2, 3, 4, 5}) == deduped);
}

TEST_CASE("unstable_unique on bidirectional iterators with custom predicate", "[utility]")
{
  std::list<int> l = {1, 1, 1, 1, 1, 2, 2, 3};
  auto new_end     = cudax::unstable_unique(l.begin(), l.end(), cuda::std::equal_to<int>{});
  REQUIRE(std::distance(l.begin(), new_end) == 3);

  std::vector<int> deduped(l.begin(), new_end);
  std::sort(deduped.begin(), deduped.end());
  REQUIRE(std::vector<int>({1, 2, 3}) == deduped);
}
