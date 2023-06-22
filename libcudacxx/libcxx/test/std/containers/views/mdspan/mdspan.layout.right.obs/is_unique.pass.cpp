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

    {
        std::layout_right::mapping<std::dextents<index_t,1>> m;

        static_assert( m.is_always_unique() == true, "" );
        assert       ( m.is_unique       () == true );
    }


    {
        std::extents<index_t,dyn,dyn> e{16, 32};
        std::layout_right::mapping<std::extents<index_t,dyn,dyn>> m{ e };

        static_assert( m.is_always_unique() == true, "" );
        assert       ( m.is_unique       () == true );
    }

    return 0;
}
