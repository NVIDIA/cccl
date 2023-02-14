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

#define CHECK_MAPPING(m) \
        assert( m.is_exhaustive()          == false); \
        assert( m.extents().rank()         == 2    ); \
        assert( m.extents().rank_dynamic() == 2    ); \
        assert( m.extents().extent(0)      == 16   ); \
        assert( m.extents().extent(1)      == 32   ); \
        assert( m.stride(0)                == 1    ); \
        assert( m.stride(1)                == 128  ); \
        assert( m.strides()[0]             == 1    ); \
        assert( m.strides()[1]             == 128  )

constexpr auto dyn = std::dynamic_extent;

int main(int, char**)
{
    // From a span
    {
        typedef int    data_t ;
        typedef size_t index_t;

        using my_ext = typename std::extents<size_t,dyn>;

        std::array<int,2> a{1, 128};
        std::span <int,2> s(a.data(), 2);
        std::layout_stride::mapping<std::extents<size_t,dyn, dyn>> m{std::dextents<size_t,2>{16, 32}, s};

        CHECK_MAPPING(m);
    }

    // TEST(TestLayoutStrideListInitialization, test_list_initialization)
    {
        typedef int    data_t ;
        typedef size_t index_t;

        std::layout_stride::mapping<std::extents<size_t,dyn, dyn>> m{std::dextents<size_t,2>{16, 32}, std::array<int,2>{1, 128}};

        CHECK_MAPPING(m);
    }

    // From another mapping
    {
        typedef int    data_t ;
        typedef size_t index_t;

        std::layout_stride::mapping<std::extents<index_t,dyn, dyn>> m0{std::dextents<index_t,2>{16, 32}, std::array<int,2>{1, 128}};
        std::layout_stride::mapping<std::extents<index_t,dyn, dyn>> m{m0};

        CHECK_MAPPING(m);
    }

    return 0;
}
