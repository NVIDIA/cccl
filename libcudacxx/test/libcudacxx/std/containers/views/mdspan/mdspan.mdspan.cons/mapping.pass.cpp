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

#include "../mdspan.mdspan.util/mdspan_util.hpp"
#include "../my_accessor.hpp"

constexpr auto dyn = cuda::std::dynamic_extent;

template <class T, class DataHandleType, class MappingType, class = void>
struct is_mapping_cons_avail : cuda::std::false_type
{};

template <class T, class DataHandleType, class MappingType>
struct is_mapping_cons_avail<
  T,
  DataHandleType,
  MappingType,
  cuda::std::enable_if_t<
    cuda::std::is_same<decltype(T{cuda::std::declval<DataHandleType>(), cuda::std::declval<MappingType>()}), T>::value>>
    : cuda::std::true_type
{};

template <class T, class DataHandleType, class MappingType>
constexpr bool is_mapping_cons_avail_v = is_mapping_cons_avail<T, DataHandleType, MappingType>::value;

int main(int, char**)
{
  using data_t    = int;
  using index_t   = size_t;
  using ext_t     = cuda::std::extents<index_t, dyn, dyn>;
  using mapping_t = cuda::std::layout_left::mapping<ext_t>;

  // mapping
  {
    using mdspan_t = cuda::std::mdspan<data_t, ext_t, cuda::std::layout_left>;

    static_assert(is_mapping_cons_avail_v<mdspan_t, int*, mapping_t> == true, "");

    cuda::std::array<data_t, 1> d{42};
    mapping_t map{cuda::std::dextents<index_t, 2>{64, 128}};
    mdspan_t m{d.data(), map};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  // Constraint: is_default_constructible_v<accessor_type> is true
  {
    using mdspan_t = cuda::std::mdspan<data_t, ext_t, cuda::std::layout_left, Foo::my_accessor<data_t>>;

    static_assert(is_mapping_cons_avail_v<mdspan_t, int*, mapping_t> == false, "");
  }

  // mapping and accessor
  {
    cuda::std::array<data_t, 1> d{42};
    mapping_t map{cuda::std::dextents<index_t, 2>{64, 128}};
    cuda::std::default_accessor<data_t> a;
    cuda::std::mdspan<data_t, ext_t, cuda::std::layout_left> m{d.data(), map, a};

    CHECK_MDSPAN_EXTENT(m, d, 64, 128);
  }

  return 0;
}
