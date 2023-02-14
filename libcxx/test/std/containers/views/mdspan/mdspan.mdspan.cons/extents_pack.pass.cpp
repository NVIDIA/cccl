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

template< class, class T, class DataHandleT, class... SizeTypes >
struct is_param_pack_cons_avail : std::false_type {};

template< class T, class DataHandleT, class... SizeTypes >
struct is_param_pack_cons_avail< std::enable_if_t< std::is_same< decltype( T{ std::declval<DataHandleT>()
                                                                            , std::declval<SizeTypes>()...
                                                                            }
                                                                         )
                                                               , T
                                                               >::value
                                                 >
                               , T
                               , DataHandleT
                               , SizeTypes...
                               > : std::true_type {};

template< class T, class DataHandleT, class... SizeTypes >
constexpr bool is_param_pack_cons_avail_v = is_param_pack_cons_avail< void, T, DataHandleT, SizeTypes... >::value;

int main(int, char**)
{
    {
        using index_t = int;
        std::array<int, 1> d{42};
        std::mdspan<int, std::extents<index_t,dyn,dyn>> m{ d.data(), 64, 128 };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    {
        using index_t = int;
        std::array<int, 1> d{42};
        std::mdspan< int
                         , std::extents<index_t,dyn,dyn>
                         , std::layout_right
                         , std::default_accessor<int>
                         > m{ d.data(), 64, 128 };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    {
        using      mdspan_t = std::mdspan< int, std::extents< int, dyn, dyn > >;
        using other_index_t = my_int;

        std::array<int, 1> d{42};
        mdspan_t m{ d.data(), other_index_t(64), other_index_t(128) };

        CHECK_MDSPAN_EXTENT(m,d,64,128);

        static_assert( is_param_pack_cons_avail_v< mdspan_t, decltype(d.data()), other_index_t, other_index_t > == true, "" );
    }

    // Constraint: (is_convertible_v<OtherIndexTypes, index_type> && ...) is true
    {
        using      mdspan_t = std::mdspan< int, std::extents< int, dyn, dyn > >;
        using other_index_t = my_int_non_convertible;

        static_assert( is_param_pack_cons_avail_v< mdspan_t, int *, other_index_t, other_index_t > == false, "" );
    }

    // Constraint: (is_nothrow_constructible<index_type, OtherIndexTypes> && ...) is true
    {
        using      mdspan_t = std::mdspan< int, std::extents< int, dyn, dyn > >;
        using other_index_t = my_int_non_nothrow_constructible;

        static_assert( is_param_pack_cons_avail_v< mdspan_t, int *, other_index_t, other_index_t > == false, "" );
    }

    // Constraint: N == rank() || N == rank_dynamic() is true
    {
        using mdspan_t = std::mdspan< int, std::extents< int, dyn, dyn > >;

        static_assert( is_param_pack_cons_avail_v< mdspan_t, int *, int > == false, "" );
    }

    // Constraint: is_constructible_v<mapping_type, extents_type> is true
    {
        using mdspan_t = std::mdspan< int, std::extents< int, 16 >, std::layout_stride >;

        static_assert( is_param_pack_cons_avail_v< mdspan_t, int *, int > == false, "" );
    }

    // Constraint: is_default_constructible_v<accessor_type> is true
    {
        using mdspan_t = std::mdspan< int, std::extents< int, 16 >, std::layout_right, Foo::my_accessor<int> >;

        static_assert( is_param_pack_cons_avail_v< mdspan_t, int *, int > == false, "" );
    }

    return 0;
}

