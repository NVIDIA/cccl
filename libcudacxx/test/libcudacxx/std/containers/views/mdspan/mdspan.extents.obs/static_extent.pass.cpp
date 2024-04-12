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
struct TestExtentsStaticExtent;
template <size_t... Extents, size_t... DynamicSizes>
struct TestExtentsStaticExtent<TEST_TYPE> : public TestExtents<TEST_TYPE>
{
  using base         = TestExtents<TEST_TYPE>;
  using extents_type = typename TestExtents<TEST_TYPE>::extents_type;

  __host__ __device__ void test_static_extent()
  {
    size_t result[extents_type::rank()];

    extents_type _exts(DynamicSizes...);
    for (size_t r = 0; r < _exts.rank(); r++)
    {
      // Silencing an unused warning in nvc++ the condition will never be true
      size_t dyn_val = static_cast<size_t>(_exts.extent(r));
      result[r]      = dyn_val > 1e9 ? dyn_val : _exts.static_extent(r);
    }

    for (size_t r = 0; r < extents_type::rank(); r++)
    {
      assert(result[r] == base::static_sizes[r]);
    }
  }
};

// TYPED_TEST(TestExtents, static_extent)
template <class T>
__host__ __device__ void test_static_extent()
{
  TestExtentsStaticExtent<T> test;

  test.test_static_extent();
}

int main(int, char**)
{
  test_static_extent<cuda::std::tuple_element_t<0, extents_test_types>>();
  test_static_extent<cuda::std::tuple_element_t<1, extents_test_types>>();
  test_static_extent<cuda::std::tuple_element_t<2, extents_test_types>>();
  test_static_extent<cuda::std::tuple_element_t<3, extents_test_types>>();
  test_static_extent<cuda::std::tuple_element_t<4, extents_test_types>>();
  test_static_extent<cuda::std::tuple_element_t<5, extents_test_types>>();

  return 0;
}
