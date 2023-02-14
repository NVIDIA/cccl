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
struct is_span_cons_avail : std::false_type {};

template< class T, class DataHandleT, class SizeType, size_t N >
struct is_span_cons_avail< T
                         , DataHandleT
                         , SizeType
                         , N
                         , std::enable_if_t< std::is_same< decltype( T{ std::declval<DataHandleT>()
                                                                      , std::declval<std::span<SizeType,N>>()
                                                                      }
                                                                   )
                                                         , T
                                                         >::value
                                           >
                         > : std::true_type {};

template< class T, class DataHandleT, class SizeType, size_t N >
constexpr bool is_span_cons_avail_v = is_span_cons_avail< T, DataHandleT, SizeType, N >::value;


int main(int, char**)
{
    // extents from std::span
    {
        using      mdspan_t = std::mdspan<int, std::extents<int,dyn,dyn>>;
        using other_index_t = int;

        static_assert( is_span_cons_avail_v< mdspan_t, int *, other_index_t, 2 > == true, "" );

        std::array<other_index_t, 1> d{42};
        std::array<other_index_t, 2> sarr{64, 128};

        mdspan_t m{d.data(), std::span<other_index_t, 2>{sarr}};

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // Constraint: (is_convertible_v<OtherIndexTypes, index_type> && ...) is true
    {
        using      mdspan_t = std::mdspan<int, std::extents<int,dyn,dyn>>;
        using other_index_t = my_int_non_convertible;

        static_assert( is_span_cons_avail_v< mdspan_t, int *, other_index_t, 2 > == false, "" );
    }

    // Constraint: (is_convertible_v<OtherIndexTypes, index_type> && ...) is true
    {
        using      mdspan_t = std::mdspan<int, std::extents<int,dyn,dyn>>;
        using other_index_t = my_int_non_convertible;

        static_assert( is_span_cons_avail_v< mdspan_t, int *, other_index_t, 2 > == false, "" );
    }

    // Constraint: N == rank() || N == rank_dynamic() is true
    {
        using      mdspan_t = std::mdspan<int, std::extents<int,dyn,dyn>>;
        using other_index_t = int;

        static_assert( is_span_cons_avail_v< mdspan_t, int *, other_index_t, 1 > == false, "" );
    }

    // Constraint: is_constructible_v<mapping_type, extents_type> is true
    {
        using mdspan_t = std::mdspan< int, std::extents< int, 16 >, std::layout_stride >;

        static_assert( is_span_cons_avail_v< mdspan_t, int *, int, 2 > == false, "" );
    }

    // Constraint: is_default_constructible_v<accessor_type> is true
    {
        using mdspan_t = std::mdspan< int, std::extents< int, 16 >, std::layout_right, Foo::my_accessor<int> >;

        static_assert( is_span_cons_avail_v< mdspan_t, int *, int, 2 > == false, "" );
    }

    return 0;
}
