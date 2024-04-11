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
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "../mdspan.layout.util/layout_util.hpp"

constexpr auto dyn = cuda::std::dynamic_extent;

template <class Extents, size_t... DynamicSizes>
using test_right_type = cuda::std::tuple<typename cuda::std::layout_right::template mapping<Extents>,
                                         cuda::std::integer_sequence<size_t, DynamicSizes...>>;

__host__ __device__ void typed_test_default_ctor_right()
{
  typed_test_default_ctor<test_right_type<cuda::std::extents<size_t, 10>>>();
  typed_test_default_ctor<test_right_type<cuda::std::extents<size_t, dyn>, 10>>();
  typed_test_default_ctor<test_right_type<cuda::std::extents<size_t, dyn, 10>, 5>>();
  typed_test_default_ctor<test_right_type<cuda::std::extents<size_t, 5, dyn>, 10>>();
  typed_test_default_ctor<test_right_type<cuda::std::extents<size_t, 5, 10>>>();
}

__host__ __device__ void typed_test_compatible_right()
{
  typed_test_compatible<test_right_type_pair<_exts<dyn>, _sizes<10>, _exts<10>, _sizes<>>>();
  typed_test_compatible<test_right_type_pair<_exts<dyn, 10>, _sizes<5>, _exts<5, dyn>, _sizes<10>>>();
  typed_test_compatible<test_right_type_pair<_exts<dyn, dyn>, _sizes<5, 10>, _exts<5, dyn>, _sizes<10>>>();
  typed_test_compatible<test_right_type_pair<_exts<dyn, dyn>, _sizes<5, 10>, _exts<dyn, 10>, _sizes<5>>>();
  typed_test_compatible<test_right_type_pair<_exts<dyn, dyn>, _sizes<5, 10>, _exts<5, 10>, _sizes<>>>();
  typed_test_compatible<test_right_type_pair<_exts<5, 10>, _sizes<>, _exts<5, dyn>, _sizes<10>>>();
  typed_test_compatible<test_right_type_pair<_exts<5, 10>, _sizes<>, _exts<dyn, 10>, _sizes<5>>>();
  typed_test_compatible<test_right_type_pair<_exts<dyn, dyn, 15>, _sizes<5, 10>, _exts<5, dyn, 15>, _sizes<10>>>();
  typed_test_compatible<test_right_type_pair<_exts<5, 10, 15>, _sizes<>, _exts<5, dyn, 15>, _sizes<10>>>();
  typed_test_compatible<test_right_type_pair<_exts<5, 10, 15>, _sizes<>, _exts<dyn, dyn, dyn>, _sizes<5, 10, 15>>>();
}

int main(int, char**)
{
  typed_test_default_ctor_right();

  typed_test_compatible_right();

  // TEST(TestLayoutRightListInitialization, test_layout_right_extent_initialization)
  {
    cuda::std::layout_right::mapping<cuda::std::extents<size_t, dyn, dyn>> m{cuda::std::dextents<size_t, 2>{16, 32}};

    static_assert(m.is_exhaustive() == true, "");
    static_assert(m.extents().rank() == 2, "");
    static_assert(m.extents().rank_dynamic() == 2, "");

    assert(m.extents().extent(0) == 16);
    assert(m.extents().extent(1) == 32);
    assert(m.stride(0) == 32);
    assert(m.stride(1) == 1);
  }

  return 0;
}
