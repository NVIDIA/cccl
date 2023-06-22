//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11

#include <mdspan>
#include <cassert>

constexpr auto dyn = std::dynamic_extent;

//template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs>

using submdspan_test_types = std::tuple<
      // LayoutLeft to LayoutLeft
      std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,1>,std::dextents<size_t,1>, std::full_extent_t>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,1>,std::dextents<size_t,1>, std::pair<int,int>>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,1>,std::dextents<size_t,0>, int>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,2>,std::dextents<size_t,2>, std::full_extent_t, std::full_extent_t>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,2>,std::dextents<size_t,2>, std::full_extent_t, std::pair<int,int>>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,2>,std::dextents<size_t,1>, std::full_extent_t, int>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,3>,std::dextents<size_t,3>, std::full_extent_t, std::full_extent_t, std::pair<int,int>>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,3>,std::dextents<size_t,2>, std::full_extent_t, std::pair<int,int>, int>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,3>,std::dextents<size_t,1>, std::full_extent_t, int, int>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,3>,std::dextents<size_t,1>, std::pair<int,int>, int, int>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,6>,std::dextents<size_t,3>, std::full_extent_t, std::full_extent_t, std::pair<int,int>, int, int, int>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,6>,std::dextents<size_t,2>, std::full_extent_t, std::pair<int,int>, int, int, int, int>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,6>,std::dextents<size_t,1>, std::full_extent_t, int, int, int ,int, int>
    , std::tuple<std::layout_left, std::layout_left, std::dextents<size_t,6>,std::dextents<size_t,1>, std::pair<int,int>, int, int, int, int, int>
    // LayoutRight to LayoutRight
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,1>,std::dextents<size_t,1>, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,1>,std::dextents<size_t,1>, std::pair<int,int>>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,1>,std::dextents<size_t,0>, int>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,2>,std::dextents<size_t,2>, std::full_extent_t, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,2>,std::dextents<size_t,2>, std::pair<int,int>, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,2>,std::dextents<size_t,1>, int, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,3>,std::dextents<size_t,3>, std::pair<int,int>, std::full_extent_t, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,3>,std::dextents<size_t,2>, int, std::pair<int,int>, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,3>,std::dextents<size_t,1>, int, int, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,6>,std::dextents<size_t,3>, int, int, int, std::pair<int,int>, std::full_extent_t, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,6>,std::dextents<size_t,2>, int, int, int, int, std::pair<int,int>, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::dextents<size_t,6>,std::dextents<size_t,1>, int, int, int, int, int, std::full_extent_t>
    // LayoutRight to LayoutRight Check Extents Preservation
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1>,std::extents<size_t,1>, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1>,std::extents<size_t,dyn>, std::pair<int,int>>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1>,std::extents<size_t>, int>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2>,std::extents<size_t,1,2>, std::full_extent_t, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2>,std::extents<size_t,dyn,2>, std::pair<int,int>, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2>,std::extents<size_t,2>, int, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2,3>,std::extents<size_t,dyn,2,3>, std::pair<int,int>, std::full_extent_t, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2,3>,std::extents<size_t,dyn,3>, int, std::pair<int,int>, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2,3>,std::extents<size_t,3>, int, int, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2,3,4,5,6>,std::extents<size_t,dyn,5,6>, int, int, int, std::pair<int,int>, std::full_extent_t, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2,3,4,5,6>,std::extents<size_t,dyn,6>, int, int, int, int, std::pair<int,int>, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_right, std::extents<size_t,1,2,3,4,5,6>,std::extents<size_t,6>, int, int, int, int, int, std::full_extent_t>

    , std::tuple<std::layout_right, std::layout_stride, std::extents<size_t,1,2,3,4,5,6>,std::extents<size_t,1,dyn,6>, std::full_extent_t, int, std::pair<int,int>, int, int, std::full_extent_t>
    , std::tuple<std::layout_right, std::layout_stride, std::extents<size_t,1,2,3,4,5,6>,std::extents<size_t,2,dyn,5>, int, std::full_extent_t, std::pair<int,int>, int, std::full_extent_t, int>
    >;

template<class T> struct TestSubMDSpan;

template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs>
struct TestSubMDSpan<
  std::tuple<LayoutOrg,
             LayoutSub,
             ExtentsOrg,
             ExtentsSub,
             SubArgs...>>
{
    using mds_org_t = std::mdspan<int, ExtentsOrg, LayoutOrg>;
    using mds_sub_t = std::mdspan<int, ExtentsSub, LayoutSub>;
    using map_t = typename mds_org_t::mapping_type;

    using mds_sub_deduced_t = decltype(std::submdspan(mds_org_t(nullptr, map_t()), SubArgs()...));
    using sub_args_t = std::tuple<SubArgs...>;
};

// TYPED_TEST(TestSubMDSpan, submdspan_return_type)
template<class T>
void test_submdspan()
{
    using TestFixture = TestSubMDSpan<T>;

    static_assert(std::is_same<typename TestFixture::mds_sub_t,
                               typename TestFixture::mds_sub_deduced_t>::value,
                  "SubMDSpan: wrong return type");
}

int main(int, char**)
{
    static_assert( std::tuple_size< submdspan_test_types >{} == 40, "" );

    test_submdspan< std::tuple_element_t<  0, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  1, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  2, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  3, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  4, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  5, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  6, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  7, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  8, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t<  9, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 10, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 11, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 12, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 13, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 14, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 15, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 16, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 17, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 18, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 19, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 20, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 21, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 22, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 23, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 24, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 25, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 26, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 27, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 28, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 29, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 30, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 31, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 32, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 33, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 34, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 35, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 36, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 37, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 38, submdspan_test_types > >();
    test_submdspan< std::tuple_element_t< 39, submdspan_test_types > >();

    return 0;
}
