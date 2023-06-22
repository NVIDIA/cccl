//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11

#include <mdspan>
#include <array>
#include <cassert>

constexpr auto dyn = std::dynamic_extent;

int main(int, char**)
{
    {
        typedef int    data_t ;
        typedef size_t index_t;

        std::array<data_t, 1> d{42};
        std::mdspan<data_t, std::extents<index_t,dyn, dyn>, std::layout_right> m{d.data(), 16, 32};

        static_assert(m.is_exhaustive() == true, "");

        assert(m.data_handle()  == d.data());
        assert(m.rank()         == 2       );
        assert(m.rank_dynamic() == 2       );
        assert(m.extent(0)      == 16      );
        assert(m.extent(1)      == 32      );
        assert(m.stride(0)      == 32      );
        assert(m.stride(1)      == 1       );
    }

    return 0;
}
