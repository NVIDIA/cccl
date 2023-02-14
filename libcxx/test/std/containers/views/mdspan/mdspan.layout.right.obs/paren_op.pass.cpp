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
#include "../mdspan.layout.util/layout_util.hpp"

constexpr auto dyn = std::dynamic_extent;

int main(int, char**)
{
    using index_t = int;

    {
        std::extents<index_t,16> e;
        std::layout_right::mapping<std::extents<index_t,16>> m{e};

        assert( m(5) == 5 );
    }

    {
        std::extents<index_t,dyn,dyn> e{16, 32};
        std::layout_right::mapping<std::extents<index_t,dyn,dyn>> m{e};

        assert( m(2,1) == 2*32 + 1*1 );
    }

    {
        std::extents<index_t,dyn,dyn,dyn> e{16, 32, 8};
        std::layout_right::mapping<std::extents<index_t,dyn,dyn,dyn>> m{e};

        assert( m(2,1,3) == 2*32*8 + 1*8 + 3*1 );
    }

    // Indices are of a type implicitly convertible to index_type
    {
        std::extents<index_t,dyn,dyn> e{16, 32};
        std::layout_right::mapping<std::extents<index_t,dyn,dyn>> m{e};

        assert( m(my_int(2),my_int(1)) == 2*32 + 1*1 );
    }

    // Constraints
    {
        std::extents<index_t,16> e;
        std::layout_right::mapping<std::extents<index_t,16>> m{e};

        unused( m );

        static_assert( is_paren_op_avail_v< decltype(m), index_t          > ==  true, "" );

        // rank consistency
        static_assert( is_paren_op_avail_v< decltype(m), index_t, index_t > == false, "" );

        // convertibility
        static_assert( is_paren_op_avail_v< decltype(m), my_int_non_convertible           > == false, "" );

        // nothrow-constructibility
        static_assert( is_paren_op_avail_v< decltype(m), my_int_non_nothrow_constructible > == false, "" );
    }

    return 0;
}
