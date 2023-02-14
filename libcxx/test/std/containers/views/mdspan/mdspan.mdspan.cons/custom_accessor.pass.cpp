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
    {
        using  data_t = int;
        using   acc_t = Foo::foo_accessor<data_t>;
        using index_t = size_t;

        std::array<data_t, 1> d{42};
        std::layout_left::mapping<std::extents<index_t,dyn,dyn>> map{std::dextents<index_t,2>{64, 128}};
        acc_t a;
        std::mdspan<data_t, std::extents<index_t,dyn,dyn>, std::layout_left, acc_t> m{ d.data(), map, a };

        static_assert(m.is_exhaustive(), "");
        //assert(m.data_handle()  == d.data());
        assert(m.rank()         == 2       );
        assert(m.rank_dynamic() == 2       );
        assert(m.extent(0)      == 64      );
        assert(m.extent(1)      == 128     );
    }

    return 0;
}
