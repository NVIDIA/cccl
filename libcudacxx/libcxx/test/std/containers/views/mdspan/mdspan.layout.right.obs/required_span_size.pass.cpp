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
        std::layout_right::mapping<std::extents<index_t,16>> m{e};

        assert( m.required_span_size() == 16 );
    }

    {
        ext2d_t e{16, 32};
        std::layout_right::mapping<ext2d_t> m{e};

        assert( m.required_span_size() == 16*32 );
    }

    {
        ext2d_t e{16, 0};
        std::layout_right::mapping<ext2d_t> m{e};

        assert( m.required_span_size() == 0 );
    }

    return 0;
}
