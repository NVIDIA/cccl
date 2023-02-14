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
#include "../mdspan.layout.util/layout_util.hpp"

constexpr auto dyn = std::dynamic_extent;

void typed_test_compare_right()
{
    typed_test_compare< test_right_type_pair<_exts<dyn         >, _sizes<10    >, _exts< 10          >, _sizes<          >> >();
    typed_test_compare< test_right_type_pair<_exts<dyn,  10    >, _sizes< 5    >, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compare< test_right_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compare< test_right_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<dyn,  10     >, _sizes< 5        >> >();
    typed_test_compare< test_right_type_pair<_exts<dyn, dyn    >, _sizes< 5, 10>, _exts<  5,  10     >, _sizes<          >> >();
    typed_test_compare< test_right_type_pair<_exts<  5,  10    >, _sizes<      >, _exts<  5, dyn     >, _sizes<10        >> >();
    typed_test_compare< test_right_type_pair<_exts<  5,  10    >, _sizes<      >, _exts<dyn,  10     >, _sizes< 5        >> >();
    typed_test_compare< test_right_type_pair<_exts<dyn, dyn, 15>, _sizes< 5, 10>, _exts<  5, dyn, 15 >, _sizes<10        >> >();
    typed_test_compare< test_right_type_pair<_exts<  5,  10, 15>, _sizes<      >, _exts<  5, dyn, 15 >, _sizes<10        >> >();
    typed_test_compare< test_right_type_pair<_exts<  5,  10, 15>, _sizes<      >, _exts<dyn, dyn, dyn>, _sizes< 5, 10, 15>> >();
}

int main(int, char**)
{
    typed_test_compare_right();

    using index_t = size_t;
    using ext1d_t = std::extents<index_t,dyn>;
    using ext2d_t = std::extents<index_t,dyn,dyn>;

    {
        ext2d_t e{64, 128};
        std::layout_right::mapping<ext2d_t> m0{  e };
        std::layout_right::mapping<ext2d_t> m { m0 };

        assert( m == m0 );
    }

    {
        ext2d_t e0{64, 128};
        ext2d_t e1{16,  32};
        std::layout_right::mapping<ext2d_t> m0{ e0 };
        std::layout_right::mapping<ext2d_t> m1{ e1 };

        assert( m0 != m1 );
    }

    return 0;
}
