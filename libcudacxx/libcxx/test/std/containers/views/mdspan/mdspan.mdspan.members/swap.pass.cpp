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

void test_std_swap_static_extents()
{
    int data1[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int data2[12] = {21,22,23,24,25,26,27,28,29,30,31,32};

    std::mdspan<int, std::extents<size_t,3,4>> m1(data1);
    std::mdspan<int, std::extents<size_t,3,4>> m2(data2);
    std::extents<size_t,3,4> exts1;
    std::layout_right::mapping<std::extents<size_t, 3, 4>> map1(exts1);
    std::extents<size_t,3,4> exts2;
    std::layout_right::mapping<std::extents<size_t, 3, 4>> map2(exts2);

    assert(m1.data_handle() == data1);
    assert(m1.mapping() == map1);
    auto val1 = m1(0,0);
    assert(val1 == 1);
    assert(m2.data_handle() == data2);
    assert(m2.mapping() == map2);
    auto val2 = m2(0,0);
    assert(val2 == 21);

    std::swap(m1,m2);
    assert(m1.data_handle() == data2);
    assert(m1.mapping() == map2);
    val1 = m1(0,0);
    assert(val1 == 21);
    assert(m2.data_handle() == data1);
    assert(m2.mapping() == map1);
    val2 = m2(0,0);
    assert(val2 == 1);
}

void test_std_swap_dynamic_extents()
{
    int data1[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int data2[12] = {21,22,23,24,25,26,27,28,29,30,31,32};

    std::mdspan<int, std::dextents<size_t,2>> m1(data1,3,4);
    std::mdspan<int, std::dextents<size_t,2>> m2(data2,4,3);
    std::dextents<size_t,2> exts1(3,4);
    std::layout_right::mapping<std::dextents<size_t,2>> map1(exts1);
    std::dextents<size_t,2> exts2(4,3);
    std::layout_right::mapping<std::dextents<size_t,2>> map2(exts2);

    assert(m1.data_handle() == data1);
    assert(m1.mapping() == map1);
    auto val1 = m1(0,0);
    assert(val1 == 1);
    assert(m2.data_handle() == data2);
    assert(m2.mapping() == map2);
    auto val2 = m2(0,0);
    assert(val2 == 21);

    std::swap(m1,m2);
    assert(m1.data_handle() == data2);
    assert(m1.mapping() == map2);
    val1 = m1(0,0);
    assert(val1 == 21);
    assert(m2.data_handle() == data1);
    assert(m2.mapping() == map1);
    val2 = m2(0,0);
    assert(val2 == 1);
}

int main(int, char**)
{
    test_std_swap_static_extents();

    test_std_swap_dynamic_extents();

    //TODO port tests for customized layout and accessor

    return 0;
}
