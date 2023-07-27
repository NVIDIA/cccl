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
#include "../mdspan.extents.util/extents_util.hpp"

// TYPED_TEST(TestExtents, copy_ctor)
template<class T>
void test_copy_con()
{
    using TestFixture = TestExtents<T>;
    TestFixture t;

    typename TestFixture::extents_type e { t.exts };
    assert(e == t.exts);
}

template< class T1, class T2, class = void >
struct is_copy_cons_avail : std::false_type {};

template< class T1, class T2 >
struct is_copy_cons_avail< T1
                         , T2
                         , std::enable_if_t< std::is_same< decltype( T1{ std::declval<T2>() } )
                                                         , T1
                                                         >::value
                                           >
                         > : std::true_type {};

template< class T1, class T2 >
constexpr bool is_copy_cons_avail_v = is_copy_cons_avail< T1, T2 >::value;

int main(int, char**)
{
    test_copy_con< std::tuple_element_t< 0, extents_test_types > >();
    test_copy_con< std::tuple_element_t< 1, extents_test_types > >();
    test_copy_con< std::tuple_element_t< 2, extents_test_types > >();
    test_copy_con< std::tuple_element_t< 3, extents_test_types > >();
    test_copy_con< std::tuple_element_t< 4, extents_test_types > >();
    test_copy_con< std::tuple_element_t< 5, extents_test_types > >();

    static_assert( is_copy_cons_avail_v< std::extents<int,2  >, std::extents<int,2> > == true , "" );

    // Constraint: rank consistency
    static_assert( is_copy_cons_avail_v< std::extents<int,2,2>, std::extents<int,2> > == false, "" );

    // Constraint: extents consistency
    static_assert( is_copy_cons_avail_v< std::extents<int,1  >, std::extents<int,2> > == false, "" );

    return 0;
}
