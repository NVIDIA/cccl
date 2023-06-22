//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11

// No CTAD in C++14 or earlier
//UNSUPPORTED: c++14

#include <mdspan>
#include <cassert>
#include "../mdspan.mdspan.util/mdspan_util.hpp"

int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
    constexpr auto dyn = std::dynamic_extent;

    // mapping
    {
        using  data_t = int;
        using index_t = size_t;
        std::array<data_t, 1> d{42};
        std::layout_left::mapping<std::extents<index_t,dyn,dyn>> map{std::dextents<index_t,2>{64, 128}};
        std::mdspan m{ d.data(), map };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    // mapping and accessor
    {
        using  data_t = int;
        using index_t = size_t;
        std::array<data_t, 1> d{42};
        std::layout_left::mapping<std::extents<index_t,dyn,dyn>> map{std::dextents<index_t,2>{64, 128}};
        std::default_accessor<data_t> a;
        std::mdspan m{ d.data(), map, a };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }
#endif

    return 0;
}
