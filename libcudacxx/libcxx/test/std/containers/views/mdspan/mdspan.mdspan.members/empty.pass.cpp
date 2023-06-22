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
    std::array<int,1> storage{1};

    {
        std::mdspan<int, std::dextents<int,1>> m;

        assert( m.empty() == true );
    }

    {
        std::mdspan<int, std::dextents<int,1>> m{ storage.data(), 0 };

        assert( m.empty() == true );
    }

    {
        std::mdspan<int, std::dextents<int,1>> m{ storage.data(), 2 };

        assert( m.empty() == false );
    }

    {
        std::mdspan<int, std::dextents<int,2>> m{ storage.data(), 2, 0 };

        assert( m.empty() == true );
    }

    {
        std::mdspan<int, std::dextents<int,2>> m{ storage.data(), 2, 2 };

        assert( m.empty() == false );
    }

    return 0;
}
