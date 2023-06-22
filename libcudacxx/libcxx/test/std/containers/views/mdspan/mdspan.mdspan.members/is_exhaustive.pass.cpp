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

        static_assert( m.is_always_exhaustive() == true, "" );
        assert       ( m.is_exhaustive       () == true );
    }

    std::array<int, 1> d{42};
    std::extents<int,dyn,dyn> e{64, 128};

    {
        std::mdspan<int, std::extents<size_t,dyn, dyn>, std::layout_left> m{ d.data(), e };

        static_assert( m.is_always_exhaustive() == true, "" );
        assert       ( m.is_exhaustive       () == true );
    }

    {
        using dexts = std::dextents<size_t,2>;

        std::mdspan< int, std::extents<size_t, dyn, dyn>, std::layout_stride >
            m { d.data()
              , std::layout_stride::template mapping<dexts>{dexts{16, 32}, std::array<std::size_t, 2>{1, 128}}
              };

        static_assert( m.is_always_exhaustive() == false, "" );
        assert       ( m.is_exhaustive       () == false );
    }

    return 0;
}

