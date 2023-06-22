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
    {
        std::mdspan<int, std::dextents<size_t,1>> m;

        static_assert( m.is_always_unique() == true, "" );
        assert       ( m.is_unique       () == true );
    }

    std::array<int, 1> d{42};
    std::extents<int,dyn,dyn> e{64, 128};

    {
        std::mdspan<int, std::extents<int,dyn,dyn>> m{ d.data(), e };

        static_assert( m.is_always_unique() == true, "" );
        assert       ( m.is_unique       () == true );
    }

    {
        std::mdspan<int, std::extents<size_t,dyn, dyn>, std::layout_left> m{ d.data(), e };

        static_assert( m.is_always_unique() == true, "" );
        assert       ( m.is_unique       () == true );
    }

    return 0;
}

