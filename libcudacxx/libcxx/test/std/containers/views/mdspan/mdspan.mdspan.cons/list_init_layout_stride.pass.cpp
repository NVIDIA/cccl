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

constexpr auto   dyn = std::dynamic_extent;

int main(int, char**)
{
    std::array<int, 1> d{42};

    std::mdspan< int
                     , std::extents<size_t, dyn, dyn>
                     , std::layout_stride
                     >
                     m { d.data()
                       , std::layout_stride::template mapping<std::dextents<size_t,2>>{std::dextents<size_t,2>{16, 32}, std::array<std::size_t, 2>{1, 128}}
                       };

    assert(m.data_handle()   == d.data());
    assert(m.rank()          == 2       );
    assert(m.rank_dynamic()  == 2       );
    assert(m.extent(0)       == 16      );
    assert(m.extent(1)       == 32      );
    assert(m.stride(0)       == 1       );
    assert(m.stride(1)       == 128     );
    assert(m.is_exhaustive() == false   );

    return 0;
}
