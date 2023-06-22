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

#define CHECK_MDSPAN(m,d) \
    static_assert(m.is_exhaustive(), ""); \
    assert(m.data_handle()  == d.data()); \
    assert(m.rank()         == 2       ); \
    assert(m.rank_dynamic() == 2       ); \
    assert(m.extent(0)      == 64      ); \
    assert(m.extent(1)      == 128     )


int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
    // TEST(TestMdspanCTAD, extents_object)
    {
        std::array<int, 1> d{42};
        std::mdspan m{d.data(), std::extents{64, 128}};

        CHECK_MDSPAN(m,d);
    }

    // TEST(TestMdspanCTAD, extents_object_move)
    {
        std::array<int, 1> d{42};
        std::mdspan m{d.data(), std::move(std::extents{64, 128})};

        CHECK_MDSPAN(m,d);
    }

    // TEST(TestMdspanCTAD, extents_std_array)
    {
        std::array<int, 1> d{42};
        std::mdspan m{d.data(), std::array{64, 128}};

        CHECK_MDSPAN(m,d);
    }

    // TEST(TestMdspanCTAD, cptr_extents_std_array)
    {
        std::array<int, 1> d{42};
        const int* const ptr= d.data();
        std::mdspan m{ptr, std::array{64, 128}};

        static_assert(std::is_same<typename decltype(m)::element_type, const int>::value, "");

        CHECK_MDSPAN(m,d);
    }

    // extents from std::span
    {
        std::array<int, 1> d{42};
        std::array<int, 2> sarr{64, 128};
        std::mdspan m{d.data(), std::span{sarr}};

        CHECK_MDSPAN(m,d);
    }
#endif

    return 0;
}
