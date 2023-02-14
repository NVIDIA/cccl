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

int main(int, char**)
{
    {
        using dexts = std::dextents<int,2>;
        std::array<int,2> a{1, 128};

        std::layout_stride::mapping<dexts> m{dexts{16, 32}, a};

        static_assert( m.is_always_strided() == true, "" );
        assert       ( m.is_strided       () == true );
    }

    return 0;
}
