//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HIERARCHY_QUERIES_H
#define SUPPORT_HIERARCHY_QUERIES_H

#include <cuda/hierarchy>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>

template <class T, class Vec>
__device__ void test_result(cuda::hierarchy_query_result<T> res, Vec exp)
{
  assert(res.x == static_cast<T>(exp.x));
  assert(res.y == static_cast<T>(exp.y));
  assert(res.z == static_cast<T>(exp.z));
}

template <class IRes, class IExp, cuda::std::size_t... Exts>
__device__ void test_result(cuda::std::extents<IRes, Exts...> res, cuda::std::extents<IExp, Exts...> exp)
{
  for (cuda::std::size_t i = 0; i < sizeof...(Exts); ++i)
  {
    assert(res.extent(i) == static_cast<IRes>(exp.extent(i)));
  }
}

template <class Level, class... Args>
__device__ void test_dims(const uint3 exp, const Level& level, Args... args)
{
  test_result(level.dims(args...), exp);
  test_result(level.template dims_as<short>(args...), exp);
  test_result(level.template dims_as<int>(args...), exp);
  test_result(level.template dims_as<long long>(args...), exp);
  test_result(level.template dims_as<unsigned short>(args...), exp);
  test_result(level.template dims_as<unsigned int>(args...), exp);
  test_result(level.template dims_as<unsigned long long>(args...), exp);
}

template <class Level, class... Args>
__device__ void test_static_dims(const ulonglong3 exp, Level level, Args... args)
{
  static_assert(level.static_dims(args...).x != 0);
  test_result(level.static_dims(args...), exp);
}

template <class Exp, class Level, class... Args>
__device__ void test_extents(const Exp exp, const Level& level, Args... args)
{
  test_result(level.extents(args...), exp);
  test_result(level.template extents_as<short>(args...), exp);
  test_result(level.template extents_as<int>(args...), exp);
  test_result(level.template extents_as<long long>(args...), exp);
  test_result(level.template extents_as<unsigned short>(args...), exp);
  test_result(level.template extents_as<unsigned int>(args...), exp);
  test_result(level.template extents_as<unsigned long long>(args...), exp);
}

template <class Level, class... Args>
__device__ void test_count(const cuda::std::size_t exp, const Level& level, Args... args)
{
  assert(level.count(args...) == exp);
  assert(level.template count_as<short>(args...) == static_cast<short>(exp));
  assert(level.template count_as<int>(args...) == static_cast<int>(exp));
  assert(level.template count_as<long long>(args...) == static_cast<long long>(exp));
  assert(level.template count_as<unsigned short>(args...) == static_cast<unsigned short>(exp));
  assert(level.template count_as<unsigned int>(args...) == static_cast<unsigned int>(exp));
  assert(level.template count_as<unsigned long long>(args...) == static_cast<unsigned long long>(exp));
}

template <class Level, class... Args>
__device__ void test_index(const uint3 exp, const Level& level, Args... args)
{
  test_result(level.index(args...), exp);
  test_result(level.template index_as<short>(args...), exp);
  test_result(level.template index_as<int>(args...), exp);
  test_result(level.template index_as<long long>(args...), exp);
  test_result(level.template index_as<unsigned short>(args...), exp);
  test_result(level.template index_as<unsigned int>(args...), exp);
  test_result(level.template index_as<unsigned long long>(args...), exp);
}

template <class Level, class... Args>
__device__ void test_rank(const cuda::std::size_t exp, const Level& level, Args... args)
{
  assert(level.rank(args...) == exp);
  assert(level.template rank_as<short>(args...) == static_cast<short>(exp));
  assert(level.template rank_as<int>(args...) == static_cast<int>(exp));
  assert(level.template rank_as<long long>(args...) == static_cast<long long>(exp));
  assert(level.template rank_as<unsigned short>(args...) == static_cast<unsigned short>(exp));
  assert(level.template rank_as<unsigned int>(args...) == static_cast<unsigned int>(exp));
  assert(level.template rank_as<unsigned long long>(args...) == static_cast<unsigned long long>(exp));
}

template <class... Args>
__device__ constexpr cuda::std::size_t mul_static_extents(Args... args)
{
  if (((args == cuda::std::dynamic_extent) || ...))
  {
    return cuda::std::dynamic_extent;
  }
  else
  {
    return (cuda::std::size_t{1} * ... * args);
  }
}

#endif // SUPPORT_HIERARCHY_QUERIES_H
