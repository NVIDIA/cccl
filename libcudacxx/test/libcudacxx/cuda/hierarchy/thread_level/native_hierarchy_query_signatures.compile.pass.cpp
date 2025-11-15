//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__hierarchy_>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

template <class Level, class T>
using HierQueryRet = cuda::hierarchy_query_result<T, cuda::std::is_same_v<Level, cuda::warp_level> ? 1 : 3>;

template <class Level>
__device__ void test_signatures(const Level& level)
{
  // 1. Test cuda::thread_level::dims(x) signature
  static_assert(cuda::std::is_same_v<HierQueryRet<Level, unsigned>, decltype(cuda::thread_level::dims(level))>);
  static_assert(noexcept(cuda::thread_level::dims(level)));

  // 2. Test cuda::thread_level::static_dims(x) signature
  static_assert(
    cuda::std::is_same_v<HierQueryRet<Level, cuda::std::size_t>, decltype(cuda::thread_level::static_dims(level))>);
  static_assert(noexcept(cuda::thread_level::static_dims(level)));

  // 3. Test cuda::thread_level::extents(x) signature
  using ExtentsRet = cuda::std::conditional_t<cuda::std::is_same_v<Level, cuda::warp_level>,
                                              cuda::std::extents<unsigned, 32>,
                                              cuda::std::dims<3, unsigned>>;
  static_assert(cuda::std::is_same_v<ExtentsRet, decltype(cuda::thread_level::extents(level))>);
  static_assert(noexcept(cuda::thread_level::extents(level)));

  // 4. Test cuda::thread_level::count(x) signature
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::thread_level::count(level))>);
  static_assert(noexcept(cuda::thread_level::count(level)));

  // 5. Test cuda::thread_level::index(x) signature
  static_assert(cuda::std::is_same_v<HierQueryRet<Level, unsigned>, decltype(cuda::thread_level::index(level))>);
  static_assert(noexcept(cuda::thread_level::index(level)));

  // 4. Test cuda::thread_level::rank(x) signature
  static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(cuda::thread_level::rank(level))>);
  static_assert(noexcept(cuda::thread_level::rank(level)));
}

__device__ void test()
{
  test_signatures(cuda::device::warp);
  test_signatures(cuda::device::block);
  test_signatures(cuda::device::cluster);
  test_signatures(cuda::device::grid);
}

int main(int, char**)
{
  return 0;
}
