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

// TYPED_TEST(TestExtents, default_ctor)
template <class T>
__host__ __device__ void test_default_con()
{
  using TestFixture = TestExtents<T>;

  auto e  = typename TestFixture::extents_type();
  auto e2 = typename TestFixture::extents_type{};
  assert(e == e2);

  for (size_t r = 0; r < e.rank(); ++r)
  {
    bool is_dynamic = (e.static_extent(r) == cuda::std::dynamic_extent);
    assert(e.extent(r) == (is_dynamic ? 0 : e.static_extent(r)));
  }
}

int main(int, char**)
{
  test_default_con<cuda::std::tuple_element_t<0, extents_test_types>>();
  test_default_con<cuda::std::tuple_element_t<1, extents_test_types>>();
  test_default_con<cuda::std::tuple_element_t<2, extents_test_types>>();
  test_default_con<cuda::std::tuple_element_t<3, extents_test_types>>();
  test_default_con<cuda::std::tuple_element_t<4, extents_test_types>>();
  test_default_con<cuda::std::tuple_element_t<5, extents_test_types>>();

  return 0;
}
