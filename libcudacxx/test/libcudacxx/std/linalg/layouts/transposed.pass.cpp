//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14
// UNSUPPORTED: msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/linalg>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename Layout, typename Map>
__host__ __device__ void map_test(Map map)
{
  using T = int;
  using E = cuda::std::extents<size_t, 2, 3>;
  cuda::std::array<T, 6> d{42, 43, 44, 45, 46, 47};
  cuda::std::mdspan<T, E, Layout> md(d.data(), map);
  auto transposed_md = cuda::std::linalg::transposed(md);

  assert(transposed_md.mapping().required_span_size() == md.mapping().required_span_size());
  assert(transposed_md.is_always_unique() == md.is_always_unique());
  assert(transposed_md.is_always_exhaustive() == md.is_always_exhaustive());
  assert(transposed_md.is_always_strided() == md.is_always_strided());
  assert(transposed_md.is_unique() == md.is_unique());
  assert(transposed_md.is_exhaustive() == md.is_exhaustive());
  assert(transposed_md.is_strided() == md.is_strided());
};

int main(int, char**)
{
  using T               = int;
  using E               = cuda::std::extents<size_t, 2, 3>;
  using dynamic_extents = cuda::std::dextents<size_t, 2>;
  // operator(), extent
  {
    cuda::std::array<T, 6> d{42, 43, 44, 45, 46, 47};
    //     42, 43, 44
    //     45, 46, 47
    cuda::std::mdspan<T, dynamic_extents> md(d.data(), 2, 3);
    auto transposed_md = cuda::std::linalg::transposed(md);

    assert(transposed_md.static_extent(0) == cuda::std::dynamic_extent);
    assert(transposed_md.static_extent(1) == cuda::std::dynamic_extent);
    assert(transposed_md.extent(0) == 3);
    assert(transposed_md.extent(1) == 2);
    assert(md(0, 0) == transposed_md(0, 0));
    assert(md(0, 1) == transposed_md(1, 0));
    assert(md(0, 2) == transposed_md(2, 0));
    assert(md(1, 0) == transposed_md(0, 1));
    assert(md(1, 1) == transposed_md(1, 1));
    assert(md(1, 2) == transposed_md(2, 1));
  }
  // required_span_size(), is_always_unique(), is_always_exhaustive(), is_always_strided(), is_unique(),
  // is_exhaustive(), is_strided()
  {
    ::map_test<cuda::std::layout_right>(cuda::std::layout_right::mapping<E>{});
    ::map_test<cuda::std::layout_left>(cuda::std::layout_left::mapping<E>{});
    ::map_test<cuda::std::layout_stride>(cuda::std::layout_stride::mapping<E>{E{}, cuda::std::array<T, 2>{10, 12}});
  }
  // stride()
  {
    cuda::std::layout_stride::mapping<E> map{E{}, cuda::std::array<T, 2>{10, 12}};
    cuda::std::array<T, 6> d{42, 43, 44, 45, 46, 47};
    cuda::std::mdspan<T, E, cuda::std::layout_stride> md(d.data(), map);
    auto transposed_md = cuda::std::linalg::transposed(md);
    assert(transposed_md.stride(0) == md.stride(1));
    assert(transposed_md.stride(1) == md.stride(0));
  }
  // constructor
  {
    using transposed_extents_t = cuda::std::extents<size_t, 3, 2>;
    cuda::std::layout_right::mapping<transposed_extents_t> map_right{};
    unused(cuda::std::linalg::layout_transpose<cuda::std::layout_right>::mapping<E>{map_right});

    cuda::std::layout_left::mapping<transposed_extents_t> map_left{};
    unused(cuda::std::linalg::layout_transpose<cuda::std::layout_left>::mapping<E>{map_left});
  }
  // operator==, operator!=
  {
    cuda::std::layout_right::mapping<dynamic_extents> map_right1{dynamic_extents{3, 2}};
    cuda::std::linalg::layout_transpose<cuda::std::layout_right>::mapping<dynamic_extents> map1{map_right1};

    cuda::std::layout_right::mapping<dynamic_extents> map_right2{dynamic_extents{3, 2}};
    cuda::std::linalg::layout_transpose<cuda::std::layout_right>::mapping<dynamic_extents> map2{map_right2};

    cuda::std::layout_right::mapping<dynamic_extents> map_right3{dynamic_extents{2, 2}};
    cuda::std::linalg::layout_transpose<cuda::std::layout_right>::mapping<dynamic_extents> map3{map_right3};

    assert(map1 == map2);
    assert(map1 != map3);
  }
  // transposed composition
  {
    cuda::std::array<T, 6> d{42, 43, 44, 45, 46, 47};
    cuda::std::mdspan<T, E> md(d.data(), E{});
    auto transposed_md1 = cuda::std::linalg::transposed(md);
    auto transposed_md2 = cuda::std::linalg::transposed(transposed_md1);
    static_assert(cuda::std::is_same_v<decltype(transposed_md2.accessor()), decltype(md.accessor())>);
    assert(transposed_md2.mapping() == md.mapping());
    assert(transposed_md2.extents() == md.extents());
    assert(md(0, 0) == transposed_md2(0, 0));
    assert(md(0, 1) == transposed_md2(0, 1));
    assert(md(0, 2) == transposed_md2(0, 2));
    assert(md(1, 0) == transposed_md2(1, 0));
    assert(md(1, 1) == transposed_md2(1, 1));
    assert(md(1, 2) == transposed_md2(1, 2));
  }
  // copy constructor
  {
    cuda::std::array<T, 6> d{42, 43, 44, 45, 46, 47};
    cuda::std::mdspan<T, E> md(d.data(), E{});
    auto transposed_md1 = cuda::std::linalg::transposed(md);
    auto transposed_md2 = transposed_md1;
    cuda::std::layout_left::mapping<dynamic_extents> map_left{dynamic_extents{3, 2}};
    assert(transposed_md2.mapping() == map_left);
    assert(transposed_md2.extent(0) == md.extent(1));
    assert(transposed_md2.extent(1) == md.extent(0));
    assert(md(0, 0) == transposed_md2(0, 0));
    assert(md(0, 1) == transposed_md2(1, 0));
    assert(md(0, 2) == transposed_md2(2, 0));
    assert(md(1, 0) == transposed_md2(0, 1));
    assert(md(1, 1) == transposed_md2(1, 1));
    assert(md(1, 2) == transposed_md2(2, 1));
  }
  return 0;
}
