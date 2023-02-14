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

template< class T, class DataHandleType, class ExtentsType, class = void >
struct is_extents_cons_avail : std::false_type {};

template< class T, class DataHandleType, class ExtentsType >
struct is_extents_cons_avail< T
                            , DataHandleType
                            , ExtentsType
                            , std::enable_if_t< std::is_same< decltype( T{ std::declval<DataHandleType>()
                                                                         , std::declval<ExtentsType>()
                                                                         }
                                                                      )
                                                            , T
                                                            >::value
                                              >
                            > : std::true_type {};

template< class T, class DataHandleType, class ExtentsType >
constexpr bool is_extents_cons_avail_v = is_extents_cons_avail< T, DataHandleType, ExtentsType >::value;

int main(int, char**)
{
    // extents from extents object
    {
        using ext_t = std::extents<int, dyn, dyn>;
        std::array<int, 1> d{42};
        std::mdspan<int, ext_t> m{ d.data(), ext_t{64, 128} };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // extents from extents object move
    {
        using    ext_t = std::extents< int, dyn, dyn >;
        using mdspan_t = std::mdspan< int, ext_t >;

        static_assert( is_extents_cons_avail_v< mdspan_t, int *, ext_t > == true, "" );

        std::array<int, 1> d{42};
        mdspan_t m{ d.data(), std::move(ext_t{64, 128}) };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // Constraint: is_constructible_v<mapping_type, extents_type> is true
    {
        using    ext_t = std::extents< int, 16, 16 >;
        using mdspan_t = std::mdspan< int, ext_t, std::layout_stride >;

        static_assert( is_extents_cons_avail_v< mdspan_t, int *, ext_t > == false, "" );
    }

    // Constraint: is_default_constructible_v<accessor_type> is true
    {
        using    ext_t = std::extents< int, 16, 16 >;
        using mdspan_t = std::mdspan< int, ext_t, std::layout_right, Foo::my_accessor<int> >;

        static_assert( is_extents_cons_avail_v< mdspan_t, int *, ext_t > == false, "" );
    }

    return 0;
}

