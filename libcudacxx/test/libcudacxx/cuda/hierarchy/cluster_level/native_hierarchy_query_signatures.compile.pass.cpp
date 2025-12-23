//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo: enable with nvrtc
// UNSUPPORTED: nvrtc

#include <cuda/hierarchy>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

template <class Level>
__device__ void test_query_signatures(const Level& level)
{
  // 1. Test cuda::cluster_level::dims(x) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<unsigned>, decltype(cuda::cluster_level::dims(level))>);
  static_assert(noexcept(cuda::cluster_level::dims(level)));

  // 2. Test cuda::cluster_level::static_dims(x) signature.
  static_assert(cuda::std::is_same_v<cuda::hierarchy_query_result<cuda::std::size_t>,
                                     decltype(cuda::cluster_level::static_dims(level))>);
  static_assert(noexcept(cuda::cluster_level::static_dims(level)));

  // 3. Test cuda::cluster_level::extents(x) signature.
  static_assert(cuda::std::is_same_v<cuda::std::dims<3, unsigned>, decltype(cuda::cluster_level::extents(level))>);
  static_assert(noexcept(cuda::cluster_level::extents(level)));

  // 4. Test cuda::cluster_level::count(x) signature.
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::cluster_level::count(level))>);
  static_assert(noexcept(cuda::cluster_level::count(level)));

  // 5. Test cuda::cluster_level::index(x) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<unsigned>, decltype(cuda::cluster_level::index(level))>);
  static_assert(noexcept(cuda::cluster_level::index(level)));

  // 6. Test cuda::cluster_level::rank(x) signature.
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::cluster_level::rank(level))>);
  static_assert(noexcept(cuda::cluster_level::rank(level)));
}

template <class T, class Level>
__device__ void test_query_as_signatures(const Level& level)
{
  // 1. Test cuda::cluster_level::dims(x) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<T>, decltype(cuda::cluster_level::dims_as<T>(level))>);
  static_assert(noexcept(cuda::cluster_level::dims_as<T>(level)));

  // 2. Test cuda::cluster_level::extents(x) signature.
  static_assert(cuda::std::is_same_v<cuda::std::dims<3, T>, decltype(cuda::cluster_level::extents_as<T>(level))>);
  static_assert(noexcept(cuda::cluster_level::extents_as<T>(level)));

  // 3. Test cuda::cluster_level::count(x) signature.
  static_assert(cuda::std::is_same_v<T, decltype(cuda::cluster_level::count_as<T>(level))>);
  static_assert(noexcept(cuda::cluster_level::count_as<T>(level)));

  // 4. Test cuda::cluster_level::index(x) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<T>, decltype(cuda::cluster_level::index_as<T>(level))>);
  static_assert(noexcept(cuda::cluster_level::index_as<T>(level)));

  // 5. Test cuda::cluster_level::rank(x) signature.
  static_assert(cuda::std::is_same_v<T, decltype(cuda::cluster_level::rank_as<T>(level))>);
  static_assert(noexcept(cuda::cluster_level::rank_as<T>(level)));
}

template <class InLevel>
__device__ void test(const InLevel& in_level)
{
  test_query_signatures(in_level);
  test_query_as_signatures<short>(in_level);
  test_query_as_signatures<int>(in_level);
  test_query_as_signatures<long long>(in_level);
  test_query_as_signatures<unsigned short>(in_level);
  test_query_as_signatures<unsigned int>(in_level);
  test_query_as_signatures<unsigned long long>(in_level);
}

__device__ void test()
{
  test(cuda::grid);
}

int main(int, char**)
{
  return 0;
}
