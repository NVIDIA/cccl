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
    // TEST(TestSubmdspanLayoutRightStaticSizedTuples, test_submdspan_layout_right_static_sized_tuples)
    {
        std::array<int,2*3*4> d;
        std::mdspan<int, std::extents<size_t,2, 3, 4>> m(d.data());
        m(1, 1, 1) = 42;
        auto sub0 = std::submdspan(m, std::tuple<int,int>{1, 2}, std::tuple<int,int>{1, 3}, std::tuple<int,int>{1, 4});

        static_assert( sub0.rank()         == 3, "" );
        static_assert( sub0.rank_dynamic() == 3, "" );
        assert( sub0.extent(0) ==  1 );
        assert( sub0.extent(1) ==  2 );
        assert( sub0.extent(2) ==  3 );
        assert( sub0(0, 0, 0)  == 42 );
    }

    return 0;
}
