//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: msvc && c++14, msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "../mdspan.extents.util/extents_util.hpp"
#include <test_macros.h>

template <class>
struct TestExtentsRank;
template <size_t... Extents, size_t... DynamicSizes>
struct TestExtentsRank<TEST_TYPE> : public TestExtents<TEST_TYPE>
{
  using base         = TestExtents<TEST_TYPE>;
  using extents_type = typename TestExtents<TEST_TYPE>::extents_type;

  __host__ __device__ void test_rank()
  {
    size_t result[2];

    extents_type _exts(DynamicSizes...);
    // Silencing an unused warning in nvc++ the condition will never be true
    size_t dyn_val = _exts.rank() > 0 ? static_cast<size_t>(_exts.extent(0)) : 1;
    result[0]      = dyn_val > 1e9 ? dyn_val : _exts.rank();
    result[1]      = _exts.rank_dynamic();

    assert(result[0] == base::static_sizes.size());
    assert(result[1] == base::dyn_sizes.size());

    // Makes sure that `rank()` returns a constexpr
    cuda::std::array<int, _exts.rank()> a;
    unused(a);
  }
};

// TYPED_TEST(TestExtents, rank)
template <class T>
__host__ __device__ void test_rank()
{
  TestExtentsRank<T> test;

  test.test_rank();
}

int main(int, char**)
{
  test_rank<cuda::std::tuple_element_t<0, extents_test_types>>();
  test_rank<cuda::std::tuple_element_t<1, extents_test_types>>();
  test_rank<cuda::std::tuple_element_t<2, extents_test_types>>();
  test_rank<cuda::std::tuple_element_t<3, extents_test_types>>();
  test_rank<cuda::std::tuple_element_t<4, extents_test_types>>();
  test_rank<cuda::std::tuple_element_t<5, extents_test_types>>();

  return 0;
}
