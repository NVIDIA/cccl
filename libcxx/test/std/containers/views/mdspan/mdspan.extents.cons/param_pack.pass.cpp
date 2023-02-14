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
#include "../my_int.hpp"

void check( std::dextents<size_t,2> e )
{
    static_assert( e.rank        () == 2, "" );
    static_assert( e.rank_dynamic() == 2, "" );

    assert( e.extent(0) == 2 );
    assert( e.extent(1) == 2 );
}

template< class, class T, class... IndexTypes >
struct is_param_pack_cons_avail : std::false_type {};

template< class T, class... IndexTypes >
struct is_param_pack_cons_avail< std::enable_if_t< std::is_same< decltype( T{ std::declval<IndexTypes>()... } )
                                                               , T
                                                               >::value
                                                 >
                               , T
                               , IndexTypes...
                               > : std::true_type {};

template< class T, class... IndexTypes >
constexpr bool is_param_pack_cons_avail_v = is_param_pack_cons_avail< void, T, IndexTypes... >::value;

int main(int, char**)
{
    {
        std::dextents<int,2> e{2, 2};

        check( e );
    }

    {
        std::dextents<int,2> e(2, 2);

        check( e );
    }

#if defined (__cpp_deduction_guides) && defined(__MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
    {
        std::extents e{2, 2};

        check( e );
    }

    {
        std::extents e(2, 2);

        check( e );
    }
#endif

    {
        std::dextents<size_t,2> e{2, 2};

        check( e );
    }

    static_assert( is_param_pack_cons_avail_v< std::dextents<int,2>, int   , int    > == true , "" );

    static_assert( is_param_pack_cons_avail_v< std::dextents<int,2>, my_int, my_int > == true , "" );

    // Constraint: rank consistency
    static_assert( is_param_pack_cons_avail_v< std::dextents<int,1>, int   , int    > == false, "" );

    // Constraint: convertibility
    static_assert( is_param_pack_cons_avail_v< std::dextents<int,1>, my_int_non_convertible           > == false, "" );

    // Constraint: nonthrow-constructibility
    static_assert( is_param_pack_cons_avail_v< std::dextents<int,1>, my_int_non_nothrow_constructible > == false, "" );

    return 0;
}

