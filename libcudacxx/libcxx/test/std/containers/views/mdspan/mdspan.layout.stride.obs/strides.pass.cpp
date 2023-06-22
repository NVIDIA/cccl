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
    using index_t = size_t;
    using ext2d_t = std::extents<index_t,dyn,dyn>;

    auto e     = std::dextents<index_t,2>{16, 32};
    auto s_arr = std::array   <index_t,2>{1, 128};

    // From a span
    {
        std::span <index_t,2> s(s_arr.data(), 2);
        std::layout_stride::mapping<ext2d_t> m{e, s};

        assert( m.strides()[0] ==   1 );
        assert( m.strides()[1] == 128 );
    }

    // From an array
    {
        std::layout_stride::mapping<ext2d_t> m{e, s_arr};

        assert( m.strides()[0] ==   1 );
        assert( m.strides()[1] == 128 );
    }

    // From another mapping
    {
        std::layout_stride::mapping<ext2d_t> m0{e, s_arr};
        std::layout_stride::mapping<ext2d_t> m{m0};

        assert( m.strides()[0] ==   1 );
        assert( m.strides()[1] == 128 );
    }

    return 0;
}

