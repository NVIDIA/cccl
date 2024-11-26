//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14
// UNSUPPORTED: msvc && c++17

#include "cuda/std/__linalg/transposed.h"

#include <cuda/std/cassert>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

int main(int, char**)
{
  // operator(), extent
  {
    cuda::std::array d{42, 43, 44, 45, 46, 47};
    //     42, 43, 44
    //     45, 46, 47
    cuda::std::mdspan md(d.data(), 2, 3);
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
    auto map_test = [](auto map) {
      cuda::std::array d{42, 43, 44, 45, 46, 47};
      cuda::std::mdspan md(d.data(), map);
      auto transposed_md = cuda::std::linalg::transposed(md);

      assert(transposed_md.mapping().required_span_size() == md.mapping().required_span_size());
      assert(transposed_md.is_always_unique() == md.is_always_unique());
      assert(transposed_md.is_always_exhaustive() == md.is_always_exhaustive());
      assert(transposed_md.is_always_strided() == md.is_always_strided());
      assert(transposed_md.is_unique() == md.is_unique());
      assert(transposed_md.is_exhaustive() == md.is_exhaustive());
      assert(transposed_md.is_strided() == md.is_strided());
    };
    using extents_t = cuda::std::extents<int, 2, 3>;
    map_test(cuda::std::layout_right::mapping<extents_t>{});
    map_test(cuda::std::layout_left::mapping<extents_t>{});
    map_test(cuda::std::layout_stride::mapping<extents_t>{extents_t{}, cuda::std::array{10, 12}});
  }
  // stride()
  {
    using extents_t = cuda::std::extents<int, 2, 3>;
    cuda::std::layout_stride::mapping<extents_t> map{extents_t{}, cuda::std::array{10, 12}};
    cuda::std::array d{42, 43, 44, 45, 46, 47};
    cuda::std::mdspan md(d.data(), map);
    auto transposed_md = cuda::std::linalg::transposed(md);
    assert(transposed_md.stride(0) == md.stride(1));
    assert(transposed_md.stride(1) == md.stride(0));
  }
  // constructor
  {
    using extents_t            = cuda::std::extents<int, 2, 3>;
    using transposed_extents_t = cuda::std::extents<int, 3, 2>;
    cuda::std::layout_right::mapping<transposed_extents_t> map_right{};
    cuda::std::linalg::layout_transpose<cuda::std::layout_right>::mapping<extents_t> map_right_transposed{map_right};
    static_cast<void>(map_right_transposed);

    cuda::std::layout_left::mapping<transposed_extents_t> map_left{};
    cuda::std::linalg::layout_transpose<cuda::std::layout_left>::mapping<extents_t> map_left_transposed{map_left};
    static_cast<void>(map_left_transposed);
  }
  // operator==
  {
    using dynamic_extents = cuda::std::dextents<int, 2>;
    cuda::std::layout_right::mapping<dynamic_extents> map_right1{dynamic_extents{3, 2}};
    cuda::std::linalg::layout_transpose<cuda::std::layout_right>::mapping<dynamic_extents> map1{map_right1};

    cuda::std::layout_right::mapping<dynamic_extents> map_right2{dynamic_extents{3, 2}};
    cuda::std::linalg::layout_transpose<cuda::std::layout_right>::mapping<dynamic_extents> map2{map_right2};

    cuda::std::layout_right::mapping<dynamic_extents> map_right3{dynamic_extents{2, 2}};
    cuda::std::linalg::layout_transpose<cuda::std::layout_right>::mapping<dynamic_extents> map3{map_right3};

    assert(map1 == map2);
    assert(!(map1 == map3));
  }
  // transposed composition
  {
    cuda::std::array d{42, 43, 44, 45, 46, 47};
    cuda::std::mdspan md(d.data(), 2, 3);
    auto transposed_md1 = cuda::std::linalg::transposed(md);
    auto transposed_md2 = cuda::std::linalg::transposed(transposed_md1);
    static_assert(cuda::std::is_same_v<decltype(transposed_md2.accessor()), decltype(md.accessor())>);
    assert(transposed_md2.mapping() == md.mapping());
    assert(transposed_md2.extents() == md.extents());
    assert(md(0, 0) == transposed_md2(0, 0));
    assert(md(0, 1) == transposed_md2(0, 1));
    assert(md(1, 0) == transposed_md2(1, 0));
    assert(md(1, 1) == transposed_md2(1, 1));
    assert(md(2, 0) == transposed_md2(2, 0));
    assert(md(2, 1) == transposed_md2(2, 1));
  }
  // copy constructor
  {
    cuda::std::array d{42, 43, 44, 45, 46, 47};
    cuda::std::mdspan md(d.data(), 2, 3);
    auto transposed_md1   = cuda::std::linalg::transposed(md);
    auto transposed_md2   = transposed_md1;
    using dynamic_extents = cuda::std::dextents<size_t, 2>;
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
