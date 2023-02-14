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
        std::extents<index_t,16,32> e;
        std::array<index_t,2> a{1,16};
        std::layout_stride::mapping<ext2d_t> m0{e, a};
        std::layout_stride::mapping<ext2d_t> m {m0};

        assert( m0 == m );
    }

    {
        using index2_t = int32_t;

        std::extents<index_t,16,32> e;
        std::array<index_t,2> a{1,16};
        std::extents<index2_t,16,32> e2;
        std::array<index2_t,2> a2{1,16};
        std::layout_stride::mapping<ext2d_t> m1{e , a };
        std::layout_stride::mapping<ext2d_t> m2{e2, a2};

        assert( m1 == m2 );
    }

    {
        std::extents<index_t,16,32> e;
        std::array<index_t,2> a0{1,16};
        std::array<index_t,2> a1{1,32};
        std::layout_stride::mapping<ext2d_t> m0{e, a0};
        std::layout_stride::mapping<ext2d_t> m1{e, a1};

        assert( m0 != m1 );
    }

    {
        std::extents<index_t,16,32> e;
        std::array<index_t,2> a{1,16};
        std::layout_stride::mapping<ext2d_t> m{e, a};
        std::layout_left  ::mapping<ext2d_t> m_left{e};

        assert( m == m_left );
    }

    {
        std::extents<index_t,16,32> e;
        std::array<index_t,2> a{32,1};
        std::layout_stride::mapping<ext2d_t> m{e, a};
        std::layout_right ::mapping<ext2d_t> m_right{e};

        assert( m == m_right );
    }

    {
        std::extents<index_t,16,32> e0;
        std::extents<index_t,16,64> e1;
        std::array<index_t,2> a{1,16};
        std::layout_stride::mapping<ext2d_t> m0{e0, a};
        std::layout_stride::mapping<ext2d_t> m1{e1, a};

        assert( m0 != m1 );
    }

    return 0;
}
