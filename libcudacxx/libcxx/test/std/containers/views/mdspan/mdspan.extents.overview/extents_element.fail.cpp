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

void check( std::dextents<size_t,2> e )
{
    static_assert( e.rank        () == 2, "" );
    static_assert( e.rank_dynamic() == 2, "" );

    assert( e.extent(0) == 2 );
    assert( e.extent(1) == 2 );
}

struct dummy {};

int main(int, char**)
{
    {
        std::dextents<int,2> e{2      , 2};

        check( e );
    }

    // Mandate: each element of Extents is either equal to dynamic_extent, or is representable as a value of type IndexType
    {

        std::dextents<int,2> e{dummy{}, 2};

        check( e );
    }

    return 0;
}
