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
#include "../mdspan.mdspan.util/mdspan_util.hpp"
#include "../my_int.hpp"
#include "../my_accessor.hpp"

constexpr auto dyn = std::dynamic_extent;

template< class T, class DataHandleT, class SizeType, size_t N, class = void >
struct is_array_cons_avail : std::false_type {};

template< class T, class DataHandleT, class SizeType, size_t N >
struct is_array_cons_avail< T
                          , DataHandleT
                          , SizeType
                          , N
                          , std::enable_if_t< std::is_same< decltype( T{ std::declval<DataHandleT>()
                                                                       , std::declval<std::array<SizeType,N>>()
                                                                       }
                                                                    )
                                                          , T
                                                          >::value
                                            >
                          > : std::true_type {};

template< class T, class DataHandleT, class SizeType, size_t N >
constexpr bool is_array_cons_avail_v = is_array_cons_avail< T, DataHandleT, SizeType, N >::value;

int main(int, char**)
{
    // extents from std::array
    {
        std::array<int, 1> d{42};
        std::mdspan<int, std::extents<int,dyn,dyn>> m{ d.data(), std::array<int, 2>{64, 128} };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // data from cptr, extents from std::array
    {
        using mdspan_t = std::mdspan<const int, std::extents<int,dyn,dyn>>;

        std::array<int, 1> d{42};
        const int* const ptr = d.data();

        static_assert( is_array_cons_avail_v< mdspan_t, decltype(ptr), int, 2 > == true, "" );

        mdspan_t m{ptr, std::array<int, 2>{64, 128}};

        static_assert( std::is_same<typename decltype(m)::element_type, const int>::value, "" );

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // Constraint: (is_convertible_v<OtherIndexTypes, index_type> && ...) is true
    {
        using      mdspan_t = std::mdspan< int, std::extents< int, dyn, dyn > >;
        using other_index_t = my_int_non_convertible;

        static_assert( is_array_cons_avail_v< mdspan_t, int *, other_index_t, 2 > == false, "" );
    }

    // Constraint: (is_nothrow_constructible<index_type, OtherIndexTypes> && ...) is true
    {
        using      mdspan_t = std::mdspan< int, std::extents< int, dyn, dyn > >;
        using other_index_t = my_int_non_nothrow_constructible;

        static_assert( is_array_cons_avail_v< mdspan_t, int *, other_index_t, 2 > == false, "" );
    }

    // Constraint: N == rank() || N == rank_dynamic() is true
    {
        using      mdspan_t = std::mdspan< int, std::extents< int, dyn, dyn > >;
        using other_index_t = int;

        static_assert( is_array_cons_avail_v< mdspan_t, int *, int, 1 > == false, "" );
    }

    // Constraint: is_constructible_v<mapping_type, extents_type> is true
    {
        using mdspan_t = std::mdspan< int, std::extents< int, 16 >, std::layout_stride >;

        static_assert( is_array_cons_avail_v< mdspan_t, int *, int, 2 > == false, "" );
    }

    // Constraint: is_default_constructible_v<accessor_type> is true
    {
        using mdspan_t = std::mdspan< int, std::extents< int, 16 >, std::layout_right, Foo::my_accessor<int> >;

        static_assert( is_array_cons_avail_v< mdspan_t, int *, int, 2 > == false, "" );
    }

    return 0;
}
