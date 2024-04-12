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

template <class>
struct TestExtentsExtent;
template <size_t... Extents, size_t... DynamicSizes>
struct TestExtentsExtent<TEST_TYPE> : public TestExtents<TEST_TYPE>
{
  using base         = TestExtents<TEST_TYPE>;
  using extents_type = typename TestExtents<TEST_TYPE>::extents_type;

  __host__ __device__ void test_extent()
  {
    size_t result[extents_type::rank()];

    extents_type _exts(DynamicSizes...);
    for (size_t r = 0; r < _exts.rank(); r++)
    {
      result[r] = _exts.extent(r);
    }

    int dyn_count = 0;
    for (size_t r = 0; r < extents_type::rank(); r++)
    {
      bool is_dynamic = base::static_sizes[r] == cuda::std::dynamic_extent;
      auto expected   = is_dynamic ? base::dyn_sizes[dyn_count++] : base::static_sizes[r];

      assert(result[r] == expected);
    }
  }
};

// TYPED_TEST(TestExtents, extent)
template <class T>
__host__ __device__ void test_extent()
{
  TestExtentsExtent<T> test;

  test.test_extent();
}

int main(int, char**)
{
  test_extent<cuda::std::tuple_element_t<0, extents_test_types>>();
  test_extent<cuda::std::tuple_element_t<1, extents_test_types>>();
  test_extent<cuda::std::tuple_element_t<2, extents_test_types>>();
  test_extent<cuda::std::tuple_element_t<3, extents_test_types>>();
  test_extent<cuda::std::tuple_element_t<4, extents_test_types>>();
  test_extent<cuda::std::tuple_element_t<5, extents_test_types>>();

  return 0;
}
