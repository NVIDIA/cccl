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
void test_span_con()
{
    using TestFixture = TestExtents<T>;
    TestFixture t;

    auto s = std::span<const size_t, t.dyn_sizes.size()>( t.dyn_sizes );
    auto e = typename TestFixture::extents_type(s);
    assert(e == t.exts);
}

template< class T, class IndexType, size_t N, class = void >
struct is_span_cons_avail : std::false_type {};

template< class T, class IndexType, size_t N >
struct is_span_cons_avail< T
                         , IndexType
                         , N
                         , std::enable_if_t< std::is_same< decltype( T{ std::declval<std::span<IndexType, N>>() } )
                                                         , T
                                                         >::value
                                           >
                         > : std::true_type {};

template< class T, class IndexType, size_t N >
constexpr bool is_span_cons_avail_v = is_span_cons_avail< T, IndexType, N >::value;

int main(int, char**)
{
    test_span_con< std::tuple_element_t< 0, extents_test_types > >();
    test_span_con< std::tuple_element_t< 1, extents_test_types > >();
    test_span_con< std::tuple_element_t< 2, extents_test_types > >();
    test_span_con< std::tuple_element_t< 3, extents_test_types > >();
    test_span_con< std::tuple_element_t< 4, extents_test_types > >();
    test_span_con< std::tuple_element_t< 5, extents_test_types > >();

    static_assert( is_span_cons_avail_v< std::dextents<int,2>, int   , 2 > == true , "" );

    static_assert( is_span_cons_avail_v< std::dextents<int,2>, my_int, 2 > == true , "" );

    // Constraint: rank consistency
    static_assert( is_span_cons_avail_v< std::dextents<int,1>, int   , 2 > == false, "" );

    // Constraint: convertibility
    static_assert( is_span_cons_avail_v< std::dextents<int,1>, my_int_non_convertible          , 1 > == false, "" );

    // Constraint: nonthrow-constructibility
    static_assert( is_span_cons_avail_v< std::dextents<int,1>, my_int_non_nothrow_constructible, 1 > == false, "" );

    return 0;
}
