//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11
// UNSUPPORTED: gcc-6

#include <mdspan>
#include <cassert>
#include "../mdspan.extents.util/extents_util.hpp"
#include "../my_int.hpp"

// TYPED_TEST(TestExtents, array_ctor)
template<class T>
void test_array_con()
{
    using TestFixture = TestExtents<T>;
    TestFixture t;

    auto e = typename TestFixture::extents_type(t.dyn_sizes);
    assert(e == t.exts);
}

template< class T, class IndexType, size_t N, class = void >
struct is_array_cons_avail : std::false_type {};

template< class T, class IndexType, size_t N >
struct is_array_cons_avail< T
                          , IndexType
                          , N
                          , std::enable_if_t< std::is_same< decltype( T{ std::declval<std::array<IndexType, N>>() } )
                                                          , T
                                                          >::value
                                            >
                          > : std::true_type {};

template< class T, class IndexType, size_t N >
constexpr bool is_array_cons_avail_v = is_array_cons_avail< T, IndexType, N >::value;

int main(int, char**)
{
    test_array_con< std::tuple_element_t< 0, extents_test_types > >();
    test_array_con< std::tuple_element_t< 1, extents_test_types > >();
    test_array_con< std::tuple_element_t< 2, extents_test_types > >();
    test_array_con< std::tuple_element_t< 3, extents_test_types > >();
    test_array_con< std::tuple_element_t< 4, extents_test_types > >();
    test_array_con< std::tuple_element_t< 5, extents_test_types > >();

    static_assert( is_array_cons_avail_v< std::dextents<   int,2>, int   , 2 > == true , "" );

    static_assert( is_array_cons_avail_v< std::dextents<   int,2>, my_int, 2 > == true , "" );

    // Constraint: rank consistency
    static_assert( is_array_cons_avail_v< std::dextents<   int,1>, int   , 2 > == false, "" );

    // Constraint: convertibility
    static_assert( is_array_cons_avail_v< std::dextents<my_int,1>, my_int_non_convertible          , 1 > == false, "" );

    // Constraint: nonthrow-constructibility
    static_assert( is_array_cons_avail_v< std::dextents<   int,1>, my_int_non_nothrow_constructible, 1 > == false, "" );

    return 0;
}
