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

    auto e     = std::dextents<index_t,2>{16, 32};
    auto s_arr = std::array   <index_t,2>{1, 128};

    // From a span
    {
        std::span <index_t,2> s(s_arr.data(), 2);
        std::layout_stride::mapping<ext2d_t> m{e, s};

        assert( m.stride(0) ==   1 );
        assert( m.stride(1) == 128 );

        static_assert( is_stride_avail_v< decltype(m), index_t > == true , "" );
    }

    // From an array
    {
        std::layout_stride::mapping<ext2d_t> m{e, s_arr};

        assert( m.stride(0) ==   1 );
        assert( m.stride(1) == 128 );
    }

    // From another mapping
    {
        std::layout_stride::mapping<ext2d_t> m0{e, s_arr};
        std::layout_stride::mapping<ext2d_t> m{m0};

        assert( m.stride(0) ==   1 );
        assert( m.stride(1) == 128 );
    }

    // constraint: extents_­type?::?rank() > 0
    {
        ext0d_t e{};
        std::layout_stride::mapping<ext0d_t> m{ e, std::array<index_t,0>{} };

        unused( m );

        static_assert( is_stride_avail_v< decltype(m), index_t > == false, "" );
    }

    return 0;
}
