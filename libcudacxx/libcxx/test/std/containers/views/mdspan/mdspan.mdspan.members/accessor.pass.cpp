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
        using  data_t = int;
        using index_t = size_t;
        std::array<data_t, 1> d{42};
        std::layout_left::mapping<std::extents<index_t,dyn,dyn>> map{std::dextents<index_t,2>{64, 128}};
        std::default_accessor<data_t> const a;
        std::mdspan<data_t, std::extents<index_t,dyn,dyn>, std::layout_left> m{ d.data(), map, a };

        assert( m.accessor().access( d.data(), 0 ) == a.access( d.data(), 0 ) );
        assert( m.accessor().offset( d.data(), 0 ) == a.offset( d.data(), 0 ) );
    }

    return 0;
}
