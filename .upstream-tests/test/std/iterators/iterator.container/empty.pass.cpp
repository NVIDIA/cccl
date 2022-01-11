//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <cuda/std/iterator>
// template <class C> constexpr auto empty(const C& c) -> decltype(c.empty());       // C++17
// template <class T, size_t N> constexpr bool empty(const T (&array)[N]) noexcept;  // C++17
// template <class E> constexpr bool empty(initializer_list<E> il) noexcept;         // C++17

#include <cuda/std/iterator>
#include <cuda/std/cassert>
#include <cuda/std/vector>
#include <cuda/std/array>
#include <cuda/std/list>
#include <cuda/std/initializer_list>

#include "test_macros.h"

#if TEST_STD_VER > 14
#include <cuda/std/string_view>
#endif

template<typename C>
void test_const_container( const C& c )
{
//  Can't say noexcept here because the container might not be
    assert ( cuda::std::empty(c)   == c.empty());
}

template<typename T>
void test_const_container( const cuda::std::initializer_list<T>& c )
{
    assert ( cuda::std::empty(c)   == (c.size() == 0));
}

template<typename C>
void test_container( C& c )
{
//  Can't say noexcept here because the container might not be
    assert ( cuda::std::empty(c)   == c.empty());
}

template<typename T>
void test_container( cuda::std::initializer_list<T>& c )
{
    ASSERT_NOEXCEPT(cuda::std::empty(c));
    assert ( cuda::std::empty(c)   == (c.size() == 0));
}

template<typename T, size_t Sz>
void test_const_array( const T (&array)[Sz] )
{
    ASSERT_NOEXCEPT(cuda::std::empty(array));
    assert (!cuda::std::empty(array));
}

int main(int, char**)
{
    cuda::std::vector<int> v; v.push_back(1);
    cuda::std::list<int>   l; l.push_back(2);
    cuda::std::array<int, 1> a; a[0] = 3;
    cuda::std::initializer_list<int> il = { 4 };

    test_container ( v );
    test_container ( l );
    test_container ( a );
    test_container ( il );

    test_const_container ( v );
    test_const_container ( l );
    test_const_container ( a );
    test_const_container ( il );

#if TEST_STD_VER > 14
    cuda::std::string_view sv{"ABC"};
    test_container ( sv );
    test_const_container ( sv );
#endif

    static constexpr int arrA [] { 1, 2, 3 };
    test_const_array ( arrA );

  return 0;
}
