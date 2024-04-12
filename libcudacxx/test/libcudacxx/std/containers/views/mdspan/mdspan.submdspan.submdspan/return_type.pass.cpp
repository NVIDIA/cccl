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

constexpr auto dyn = cuda::std::dynamic_extent;

// template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs>

using submdspan_test_types = cuda::std::tuple<
  // LayoutLeft to LayoutLeft
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::pair<int, int>>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::dextents<size_t, 0>,
                   int>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::full_extent_t,
                   cuda::std::pair<int, int>>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::full_extent_t,
                   int>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t,
                   cuda::std::pair<int, int>>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::full_extent_t,
                   cuda::std::pair<int, int>,
                   int>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::full_extent_t,
                   int,
                   int>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::pair<int, int>,
                   int,
                   int>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 6>,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t,
                   cuda::std::pair<int, int>,
                   int,
                   int,
                   int>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 6>,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::full_extent_t,
                   cuda::std::pair<int, int>,
                   int,
                   int,
                   int,
                   int>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 6>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::full_extent_t,
                   int,
                   int,
                   int,
                   int,
                   int>,
  cuda::std::tuple<cuda::std::layout_left,
                   cuda::std::layout_left,
                   cuda::std::dextents<size_t, 6>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::pair<int, int>,
                   int,
                   int,
                   int,
                   int,
                   int>
  // LayoutRight to LayoutRight
  ,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::pair<int, int>>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 1>,
                   cuda::std::dextents<size_t, 0>,
                   int>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 2>,
                   cuda::std::dextents<size_t, 1>,
                   int,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::dextents<size_t, 2>,
                   int,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 3>,
                   cuda::std::dextents<size_t, 1>,
                   int,
                   int,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 6>,
                   cuda::std::dextents<size_t, 3>,
                   int,
                   int,
                   int,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 6>,
                   cuda::std::dextents<size_t, 2>,
                   int,
                   int,
                   int,
                   int,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::dextents<size_t, 6>,
                   cuda::std::dextents<size_t, 1>,
                   int,
                   int,
                   int,
                   int,
                   int,
                   cuda::std::full_extent_t>
  // LayoutRight to LayoutRight Check Extents Preservation
  ,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1>,
                   cuda::std::extents<size_t, 1>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1>,
                   cuda::std::extents<size_t, dyn>,
                   cuda::std::pair<int, int>>,
  cuda::std::
    tuple<cuda::std::layout_right, cuda::std::layout_right, cuda::std::extents<size_t, 1>, cuda::std::extents<size_t>, int>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2>,
                   cuda::std::extents<size_t, 1, 2>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2>,
                   cuda::std::extents<size_t, dyn, 2>,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2>,
                   cuda::std::extents<size_t, 2>,
                   int,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2, 3>,
                   cuda::std::extents<size_t, dyn, 2, 3>,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2, 3>,
                   cuda::std::extents<size_t, dyn, 3>,
                   int,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2, 3>,
                   cuda::std::extents<size_t, 3>,
                   int,
                   int,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2, 3, 4, 5, 6>,
                   cuda::std::extents<size_t, dyn, 5, 6>,
                   int,
                   int,
                   int,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2, 3, 4, 5, 6>,
                   cuda::std::extents<size_t, dyn, 6>,
                   int,
                   int,
                   int,
                   int,
                   cuda::std::pair<int, int>,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_right,
                   cuda::std::extents<size_t, 1, 2, 3, 4, 5, 6>,
                   cuda::std::extents<size_t, 6>,
                   int,
                   int,
                   int,
                   int,
                   int,
                   cuda::std::full_extent_t>

  ,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_stride,
                   cuda::std::extents<size_t, 1, 2, 3, 4, 5, 6>,
                   cuda::std::extents<size_t, 1, dyn, 6>,
                   cuda::std::full_extent_t,
                   int,
                   cuda::std::pair<int, int>,
                   int,
                   int,
                   cuda::std::full_extent_t>,
  cuda::std::tuple<cuda::std::layout_right,
                   cuda::std::layout_stride,
                   cuda::std::extents<size_t, 1, 2, 3, 4, 5, 6>,
                   cuda::std::extents<size_t, 2, dyn, 5>,
                   int,
                   cuda::std::full_extent_t,
                   cuda::std::pair<int, int>,
                   int,
                   cuda::std::full_extent_t,
                   int>>;

template <class T>
struct TestSubMDSpan;

template <class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class... SubArgs>
struct TestSubMDSpan<cuda::std::tuple<LayoutOrg, LayoutSub, ExtentsOrg, ExtentsSub, SubArgs...>>
{
  using mds_org_t = cuda::std::mdspan<int, ExtentsOrg, LayoutOrg>;
  using mds_sub_t = cuda::std::mdspan<int, ExtentsSub, LayoutSub>;
  using map_t     = typename mds_org_t::mapping_type;

  using mds_sub_deduced_t = decltype(cuda::std::submdspan(mds_org_t(nullptr, map_t()), SubArgs()...));
  using sub_args_t        = cuda::std::tuple<SubArgs...>;
};

// TYPED_TEST(TestSubMDSpan, submdspan_return_type)
template <class T>
__host__ __device__ void test_submdspan()
{
  using TestFixture = TestSubMDSpan<T>;

  static_assert(cuda::std::is_same<typename TestFixture::mds_sub_t, typename TestFixture::mds_sub_deduced_t>::value,
                "SubMDSpan: wrong return type");
}

int main(int, char**)
{
  static_assert(cuda::std::tuple_size<submdspan_test_types>{} == 40, "");

  test_submdspan<cuda::std::tuple_element_t<0, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<1, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<2, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<3, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<4, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<5, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<6, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<7, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<8, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<9, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<10, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<11, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<12, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<13, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<14, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<15, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<16, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<17, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<18, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<19, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<20, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<21, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<22, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<23, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<24, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<25, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<26, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<27, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<28, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<29, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<30, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<31, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<32, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<33, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<34, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<35, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<36, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<37, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<38, submdspan_test_types>>();
  test_submdspan<cuda::std::tuple_element_t<39, submdspan_test_types>>();

  return 0;
}
