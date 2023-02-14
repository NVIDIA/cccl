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
        std::array<index_t,1> a{1};
        std::layout_stride::mapping<std::extents<index_t,16>> m{e, a};

        assert( m(8) == 8 );
    }

    {
        std::extents<index_t,dyn,dyn> e{16, 32};
        std::array<index_t,2> a{1,16};
        std::layout_stride::mapping<std::extents<index_t,dyn,dyn>> m{e, a};

        assert( m(8,16) == 8*1 + 16*16 );
    }

    {
        std::extents<index_t,16,dyn> e{32};
        std::array<index_t,2> a{1,24};
        std::layout_stride::mapping<std::extents<index_t,dyn,dyn>> m{e, a};

        assert( m(8,16) == 8*1 + 16*24 );
    }

    {
        std::extents<index_t,16,dyn> e{32};
        std::array<index_t,2> a{48,1};
        std::layout_stride::mapping<std::extents<index_t,dyn,dyn>> m{e, a};

        assert( m(8,16) == 8*48 + 16*1 );
    }

    // Indices are of a type implicitly convertible to index_type
    {
        std::extents<index_t,dyn,dyn> e{16, 32};
        std::array<index_t,2> a{1,16};
        std::layout_stride::mapping<std::extents<index_t,dyn,dyn>> m{e, a};

        assert( m(my_int(8),my_int(16)) == 8*1 + 16*16 );
    }

    // Constraints
    {
        std::extents<index_t,16> e;
        std::array<index_t,1> a{1};
        std::layout_stride::mapping<std::extents<index_t,16>> m{e, a};

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

