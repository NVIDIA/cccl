//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/mdspan>
#include <cuda/std/tuple>

#define TEST_TYPE \
  cuda::std::tuple<cuda::std::extents<size_t, Extents...>, cuda::std::integer_sequence<size_t, DynamicSizes...>>

template <class>
struct TestExtents;
template <size_t... Extents, size_t... DynamicSizes>
struct TestExtents<
  cuda::std::tuple<cuda::std::extents<size_t, Extents...>, cuda::std::integer_sequence<size_t, DynamicSizes...>>>
{
  using extents_type = cuda::std::extents<size_t, Extents...>;
  // Double Braces here to make it work with GCC 5
  // Otherwise: "error: array must be initialized with a brace-enclosed initializer"
  const cuda::std::array<size_t, sizeof...(Extents)> static_sizes{{Extents...}};
  const cuda::std::array<size_t, sizeof...(DynamicSizes)> dyn_sizes{{DynamicSizes...}};
  extents_type exts{DynamicSizes...};
};

template <size_t... Ds>
using _sizes = cuda::std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using _exts = cuda::std::extents<size_t, Ds...>;

constexpr auto dyn = cuda::std::dynamic_extent;

using extents_test_types =
  cuda::std::tuple<cuda::std::tuple<_exts<10>, _sizes<>>,
                   cuda::std::tuple<_exts<dyn>, _sizes<10>>,
                   cuda::std::tuple<_exts<10, 3>, _sizes<>>,
                   cuda::std::tuple<_exts<dyn, 3>, _sizes<10>>,
                   cuda::std::tuple<_exts<10, dyn>, _sizes<3>>,
                   cuda::std::tuple<_exts<dyn, dyn>, _sizes<10, 3>>>;
