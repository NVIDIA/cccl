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
    using ext1d_t = std::extents<index_t,dyn>;
    using ext2d_t = std::extents<index_t,dyn,dyn>;

    {
        ext2d_t e{16, 32};
        std::layout_left::mapping<ext2d_t> m{e};

        assert( m.extents() == e );
    }

    {
        ext1d_t e{16};
        std::layout_right::mapping<ext1d_t> m_right{e};
        std::layout_left ::mapping<ext1d_t> m{m_right};

        assert( m.extents() == e );
    }

    {
        ext2d_t e{16, 32};
        std::array<index_t,2> a{1,16};
        std::layout_stride::mapping<ext2d_t> m_stride{e, a};
        std::layout_left  ::mapping<ext2d_t> m{m_stride};

        assert( m.extents() == e );
    }

    return 0;
}
