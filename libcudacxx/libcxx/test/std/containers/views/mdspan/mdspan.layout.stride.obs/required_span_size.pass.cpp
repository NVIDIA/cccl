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

constexpr auto dyn = std::dynamic_extent;

int main(int, char**)
{
    using index_t = int;
    using ext2d_t = std::extents<index_t,dyn,dyn>;

    {
        std::extents<index_t,16> e;
        std::array<index_t,1> a{1};
        std::layout_stride::mapping<std::extents<index_t,16>> m{e, a};

        assert( m.required_span_size() == 16 );
    }

    {
        ext2d_t e{16, 32};
        std::array<index_t,2> a{1,16};
        std::layout_stride::mapping<ext2d_t> m{e, a};

        assert( m.required_span_size() == 16*32 );
    }

    {
        ext2d_t e{16, 0};
        std::array<index_t,2> a{1,1};
        std::layout_stride::mapping<ext2d_t> m{e, a};

        assert( m.required_span_size() == 0 );
    }

    {
        std::extents<index_t,16,dyn> e{32};
        std::array<index_t,2> a{1,24};
        std::layout_stride::mapping<ext2d_t> m{e, a};

        assert( m.required_span_size() == 32*24 - (24-16) );
    }

    {
        std::extents<index_t,16,dyn> e{32};
        std::array<index_t,2> a{48,1};
        std::layout_stride::mapping<ext2d_t> m{e, a};

        assert( m.required_span_size() == 16*48 - (48-32) );
    }

    return 0;
}
