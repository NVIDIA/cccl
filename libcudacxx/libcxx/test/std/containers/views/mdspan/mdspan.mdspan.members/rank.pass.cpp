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
    typedef int    data_t ;
    typedef size_t index_t;

    std::array<data_t, 1> d{42};

    {
        std::mdspan<data_t, std::dextents<index_t,1>> m;

        static_assert( m.rank        () == 1, "" );
        assert       ( m.rank_dynamic() == 1 );
    }

    {
        std::mdspan<data_t, std::extents<index_t, 16>> m{d.data()};

        static_assert( m.rank        () == 1, "" );
        assert       ( m.rank_dynamic() == 0 );
    }

    {
        std::mdspan<data_t, std::extents<index_t, dyn, dyn>> m{d.data(), 16, 32};

        static_assert( m.rank        () == 2, "" );
        assert       ( m.rank_dynamic() == 2 );
    }

    {
        std::mdspan<data_t, std::extents<index_t, 8, dyn, dyn>> m{d.data(), 16, 32};

        static_assert( m.rank        () == 3, "" );
        assert       ( m.rank_dynamic() == 2 );
    }

    return 0;
}
