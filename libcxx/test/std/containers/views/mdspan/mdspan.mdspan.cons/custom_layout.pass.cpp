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
#include "../mdspan.mdspan.util/mdspan_util.hpp"
#include "../foo_customizations.hpp"

constexpr auto dyn = std::dynamic_extent;

int main(int, char**)
{
    using map_t = Foo::layout_foo::template mapping<std::dextents<size_t ,2>>;

    {
        using  data_t = int;
        using   lay_t = Foo::layout_foo;
        using index_t = size_t;

        std::array<data_t, 1> d{42};
        lay_t::mapping<std::extents<index_t,dyn,dyn>> map{std::dextents<index_t,2>{64, 128}};
        std::mdspan<data_t, std::extents<index_t,dyn,dyn>, lay_t> m{ d.data(), map };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    return 0;
}
