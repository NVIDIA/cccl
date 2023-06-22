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
    static_assert(std::is_same<decltype(m)::element_type,int>::value, ""); \
    static_assert(m.is_exhaustive(), ""); \
    assert(m.data_handle()  == d.data()); \
    assert(m.rank()         == 0       ); \
    assert(m.rank_dynamic() == 0       )

int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
    // TEST(TestMdspanCTAD, ctad_pointer)
    {
        std::array<int,5> d = {1,2,3,4,5};
        int* ptr = d.data();
        std::mdspan m(ptr);

        CHECK_MDSPAN(m,d);
    }

    // TEST(TestMdspanCTAD, ctad_pointer_tmp)
    {
        std::array<int,5> d = {1,2,3,4,5};
        std::mdspan m(d.data());

        CHECK_MDSPAN(m,d);
    }

    // TEST(TestMdspanCTAD, ctad_pointer_move)
    {
        std::array<int,5> d = {1,2,3,4,5};
        int* ptr = d.data();
        std::mdspan m(std::move(ptr));

        CHECK_MDSPAN(m,d);
    }
#endif

    return 0;
}
