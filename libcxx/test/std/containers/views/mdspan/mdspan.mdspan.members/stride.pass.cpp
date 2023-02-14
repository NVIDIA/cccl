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

        assert( m.stride(0) == 1 );
        assert( m.stride(1) == 1 );
    }

    {
        std::mdspan<data_t, std::extents<index_t, dyn, dyn>> m{d.data(), 16, 32};

        assert( m.stride(0) == 32 );
        assert( m.stride(1) == 1  );
    }

    {
        std::array<data_t, 1> d{42};
        std::mdspan<data_t, std::extents<index_t, dyn, dyn>, std::layout_left> m{d.data(), 16, 32};

        assert( m.stride(0) == 1  );
        assert( m.stride(1) == 16 );
    }

    {
        using dexts = std::dextents<size_t,2>;

        std::mdspan< int, std::extents<size_t, dyn, dyn>, std::layout_stride >
            m { d.data()
              , std::layout_stride::template mapping<dexts>{dexts{16, 32}, std::array<std::size_t, 2>{1, 128}}
              };

        assert( m.stride(0) == 1   );
        assert( m.stride(1) == 128 );
    }

    return 0;
}

