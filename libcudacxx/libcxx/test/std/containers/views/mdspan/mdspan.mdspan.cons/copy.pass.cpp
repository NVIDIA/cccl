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
#include "../my_accessor.hpp"

constexpr auto dyn = std::dynamic_extent;

template< class T1, class T0, class = void >
struct is_copy_cons_avail : std::false_type {};

template< class T1, class T0 >
struct is_copy_cons_avail< T1
                         , T0
                         , std::enable_if_t< std::is_same< decltype( T1{ std::declval<T0>() } ), T1 >::value >
                         > : std::true_type {};

template< class T1, class T0 >
constexpr bool is_copy_cons_avail_v = is_copy_cons_avail< T1, T0 >::value;

int main(int, char**)
{
    // copy constructor
    {
        using ext_t    = std::extents<int, dyn, dyn>;
        using mdspan_t = std::mdspan<int, ext_t>;

        static_assert( is_copy_cons_avail_v< mdspan_t, mdspan_t > == true, "" );

        std::array<int, 1> d{42};
        mdspan_t m0{ d.data(), ext_t{64, 128} };
        mdspan_t m { m0 };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // copy constructor with conversion
    {
        std::array<int, 1> d{42};
        std::mdspan<int, std::extents<int,64,128>> m0{ d.data(), std::extents<int,64,128>{} };
        std::mdspan<const int, std::extents<size_t,dyn,dyn>> m { m0 };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // Constraint: is_constructible_v<mapping_type, const OtherLayoutPolicy::template mapping<OtherExtents>&> is true
    {
        using mdspan1_t = std::mdspan<int, std::extents<int,dyn,dyn>, std::layout_left >;
        using mdspan0_t = std::mdspan<int, std::extents<int,dyn,dyn>, std::layout_right>;

        static_assert( is_copy_cons_avail_v< mdspan1_t, mdspan0_t > == false, "" );
    }

    // Constraint: is_constructible_v<accessor_type, const OtherAccessor&> is true
    {
        using mdspan1_t = std::mdspan<int, std::extents<int,dyn,dyn>, std::layout_right, Foo::my_accessor<int>>;
        using mdspan0_t = std::mdspan<int, std::extents<int,dyn,dyn>, std::layout_right>;

        static_assert( is_copy_cons_avail_v< mdspan1_t, mdspan0_t > == false, "" );
    }

    return 0;
}
