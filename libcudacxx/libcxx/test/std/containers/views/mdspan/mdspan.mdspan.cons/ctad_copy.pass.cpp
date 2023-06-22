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

    // copy constructor
    {
        std::array<int, 1> d{42};
        std::mdspan<int, std::extents<size_t,dyn,dyn>> m0{ d.data(), std::extents{64, 128} };
        std::mdspan m{ m0 };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }
#endif

    return 0;
}
