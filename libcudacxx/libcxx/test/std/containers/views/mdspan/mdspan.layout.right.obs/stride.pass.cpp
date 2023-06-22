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
#include "../mdspan.layout.util/layout_util.hpp"

constexpr auto dyn = std::dynamic_extent;

int main(int, char**)
{
    using index_t = size_t;
    using ext0d_t = std::extents<index_t>;
    using ext2d_t = std::extents<index_t,dyn,dyn>;

    {
        ext2d_t e{64, 128};
        std::layout_right::mapping<ext2d_t> m{ e };

        assert( m.stride(0) == 128 );
        assert( m.stride(1) ==   1 );

        static_assert( is_stride_avail_v< decltype(m), index_t > == true , "" );
    }

    {
        ext2d_t e{64, 1};
        std::layout_right::mapping<ext2d_t> m{ e };

        assert( m.stride(0) == 1 );
        assert( m.stride(1) == 1 );
    }

    // constraint: extents_type::rank() > 0
    {
        ext0d_t e{};
        std::layout_right::mapping<ext0d_t> m{ e };

        unused( m );

        static_assert( is_stride_avail_v< decltype(m), index_t > == false, "" );
    }

    return 0;
}
