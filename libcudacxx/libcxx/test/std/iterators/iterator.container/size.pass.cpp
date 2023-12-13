//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <iterator>
// template <class C> constexpr auto size(const C& c) -> decltype(c.size());         // C++17
// template <class T, size_t N> constexpr size_t size(const T (&array)[N]) noexcept; // C++17

#include <iterator>
#include <cassert>
#include <vector>
#include <array>
#include <list>
#include <initializer_list>

#include "test_macros.h"

#if TEST_STD_VER > 2014
#include <string_view>
#endif


template<typename C>
void test_container( C& c)
{
//  Can't say noexcept here because the container might not be
    assert ( std::size(c)   == c.size());
}

template<typename C>
void test_const_container( const C& c )
{
//  Can't say noexcept here because the container might not be
    assert ( std::size(c)   == c.size());
}

template<typename T>
void test_const_container( const std::initializer_list<T>& c)
{
    LIBCPP_ASSERT_NOEXCEPT(std::size(c)); // our std::size is conditionally noexcept
    assert ( std::size(c)   == c.size());
}

template<typename T>
void test_container( std::initializer_list<T>& c )
{
    LIBCPP_ASSERT_NOEXCEPT(std::size(c)); // our std::size is conditionally noexcept
    assert ( std::size(c)   == c.size());
}

template<typename T, size_t Sz>
void test_const_array( const T (&array)[Sz] )
{
    ASSERT_NOEXCEPT(std::size(array));
    assert ( std::size(array) == Sz );
}

int main(int, char**)
{
    std::vector<int> v; v.push_back(1);
    std::list<int>   l; l.push_back(2);
    std::array<int, 1> a; a[0] = 3;
    std::initializer_list<int> il = { 4 };
    test_container ( v );
    test_container ( l );
    test_container ( a );
    test_container ( il );

    test_const_container ( v );
    test_const_container ( l );
    test_const_container ( a );
    test_const_container ( il );

#if TEST_STD_VER > 2014
    std::string_view sv{"ABC"};
    test_container ( sv );
    test_const_container ( sv );
#endif

    static constexpr int arrA [] { 1, 2, 3 };
    test_const_array ( arrA );

  return 0;
}
