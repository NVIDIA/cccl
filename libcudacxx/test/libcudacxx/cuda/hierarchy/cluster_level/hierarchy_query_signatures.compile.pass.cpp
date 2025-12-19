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
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

template <class Level, class Hierarchy>
__device__ void test_query_signatures(const Level& level, const Hierarchy& hier)
{
  // 1. Test cuda::cluster_level::dims(x, hier) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<unsigned>, decltype(cuda::cluster_level::dims(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::dims(level, hier)));

  // 2. Test cuda::cluster_level::static_dims(x, hier) signature.
  static_assert(cuda::std::is_same_v<cuda::hierarchy_query_result<cuda::std::size_t>,
                                     decltype(cuda::cluster_level::static_dims(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::static_dims(level, hier)));

  // 3. Test cuda::cluster_level::extents(x, hier) signature.
  using ExtentsResult = decltype(cuda::cluster_level::extents(level, hier));
  static_assert(cuda::std::__is_cuda_std_extents_v<ExtentsResult>);
  static_assert(cuda::std::is_same_v<unsigned, typename ExtentsResult::index_type>);
  static_assert(noexcept(cuda::cluster_level::extents(level, hier)));

  // 4. Test cuda::cluster_level::count(x, hier) signature.
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::cluster_level::count(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::count(level, hier)));

  // 5. Test cuda::cluster_level::index(x, hier) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<unsigned>, decltype(cuda::cluster_level::index(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::index(level, hier)));

  // 6. Test cuda::cluster_level::rank(x, hier) signature.
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::cluster_level::rank(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::rank(level, hier)));
}

template <class T, class Level, class Hierarchy>
__device__ void test_query_as_signatures(const Level& level, const Hierarchy& hier)
{
  // 1. Test cuda::cluster_level::dims_as(x, hier) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<T>, decltype(cuda::cluster_level::dims_as<T>(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::dims_as<T>(level, hier)));

  // 2. Test cuda::cluster_level::extents_as(x, hier) signature.
  using ExtentsResult = decltype(cuda::cluster_level::extents_as<T>(level, hier));
  static_assert(cuda::std::__is_cuda_std_extents_v<ExtentsResult>);
  static_assert(cuda::std::is_same_v<T, typename ExtentsResult::index_type>);
  static_assert(noexcept(cuda::cluster_level::extents_as<T>(level, hier)));

  // 3. Test cuda::cluster_level::count_as(x, hier) signature.
  static_assert(cuda::std::is_same_v<T, decltype(cuda::cluster_level::count_as<T>(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::count_as<T>(level, hier)));

  // 4. Test cuda::cluster_level::index_as(x, hier) signature.
  static_assert(
    cuda::std::is_same_v<cuda::hierarchy_query_result<T>, decltype(cuda::cluster_level::index_as<T>(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::index_as<T>(level, hier)));

  // 5. Test cuda::cluster_level::rank_as(x, hier) signature.
  static_assert(cuda::std::is_same_v<T, decltype(cuda::cluster_level::rank_as<T>(level, hier))>);
  static_assert(noexcept(cuda::cluster_level::rank_as<T>(level, hier)));
}

template <class InLevel, class Hierarchy>
__device__ void test(const InLevel& in_level, const Hierarchy& hier)
{
  test_query_signatures(in_level, hier);
  test_query_as_signatures<short>(in_level, hier);
  test_query_as_signatures<int>(in_level, hier);
  test_query_as_signatures<long long>(in_level, hier);
  test_query_as_signatures<unsigned short>(in_level, hier);
  test_query_as_signatures<unsigned int>(in_level, hier);
  test_query_as_signatures<unsigned long long>(in_level, hier);
}

template <class Hierarchy>
__device__ void test(const Hierarchy& hier)
{
  test(cuda::grid, hier);
}

template <class Hierarchy>
__global__ void test_kernel(Hierarchy hier)
{
  test(hier);
}

#define TEST_KERNEL_INSTANTIATE(...)                                                 \
  template __global__ void test_kernel<decltype(cuda::make_hierarchy(__VA_ARGS__))>( \
    decltype(cuda::make_hierarchy(__VA_ARGS__)))

TEST_KERNEL_INSTANTIATE(cuda::grid_dims<1>(), cuda::cluster_dims<1>(), cuda::block_dims<1>());
TEST_KERNEL_INSTANTIATE(cuda::grid_dims<1>(), cuda::cluster_dims<1>(), cuda::block_dims(dim3{}));
TEST_KERNEL_INSTANTIATE(cuda::grid_dims<1>(), cuda::cluster_dims(dim3{}), cuda::block_dims<1>());
TEST_KERNEL_INSTANTIATE(cuda::grid_dims<1>(), cuda::cluster_dims(dim3{}), cuda::block_dims(dim3{}));
TEST_KERNEL_INSTANTIATE(cuda::grid_dims(dim3{}), cuda::cluster_dims<1>(), cuda::block_dims<1>());
TEST_KERNEL_INSTANTIATE(cuda::grid_dims(dim3{}), cuda::cluster_dims<1>(), cuda::block_dims(dim3{}));
TEST_KERNEL_INSTANTIATE(cuda::grid_dims(dim3{}), cuda::cluster_dims(dim3{}), cuda::block_dims<1>());
TEST_KERNEL_INSTANTIATE(cuda::grid_dims(dim3{}), cuda::cluster_dims(dim3{}), cuda::block_dims(dim3{}));

int main(int, char**)
{
  return 0;
}
