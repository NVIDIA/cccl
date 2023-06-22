//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mdspan>
#include <tuple>

#define TEST_TYPE \
    std::tuple < \
        std::extents<size_t, Extents...>, \
        std::integer_sequence<size_t, DynamicSizes...> \
    >


template <class> struct TestExtents;
template <size_t... Extents, size_t... DynamicSizes>
struct TestExtents<
    std::tuple<
        std::extents<size_t, Extents...>,
        std::integer_sequence<size_t, DynamicSizes...>
    >
>
{
    using extents_type = std::extents<size_t,Extents...>;
    // Double Braces here to make it work with GCC 5
    // Otherwise: "error: array must be initialized with a brace-enclosed initializer"
    const std::array<size_t, sizeof...(Extents)> static_sizes {{ Extents... }};
    const std::array<size_t, sizeof...(DynamicSizes)> dyn_sizes {{ DynamicSizes... }};
    extents_type exts { DynamicSizes... };
};

template <size_t... Ds>
using _sizes = std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using _exts = std::extents<size_t,Ds...>;

constexpr auto dyn = std::dynamic_extent;

using extents_test_types = std::tuple<
    std::tuple< _exts< 10     >, _sizes<     > >
  , std::tuple< _exts<dyn     >, _sizes<10   > >
  , std::tuple< _exts< 10,   3>, _sizes<     > >
  , std::tuple< _exts<dyn,   3>, _sizes<10   > >
  , std::tuple< _exts< 10, dyn>, _sizes< 3   > >
  , std::tuple< _exts<dyn, dyn>, _sizes<10, 3> >
>;
