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
        std::array<int, 1> d{42};
        std::extents<int,dyn,dyn> e{64, 128};
        std::mdspan<int, std::extents<int,dyn,dyn>> m{ d.data(), e };

        assert( &m.extents() == &m.mapping().extents() );
    }

    return 0;
}

